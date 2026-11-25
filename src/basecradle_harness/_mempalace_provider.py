"""The MemPalace adapter: files each memory under a BaseCradle name, keeps its own record, stamps nothing.

Selected by ``HARNESS_MEMORY_PROVIDER=mempalace`` (issue #1011, decided by ``@origin`` on 2026-11-23
with ``@briggs``'s review folded in; the record is basecradle#966). The adapter the fleet ran before
it stays selectable, unchanged, as ``mempalace-previous`` (`_mempalace.MemPalaceMemoryProvider`):
cutover sets an agent's setting to ``mempalace``, rollback sets it back, and both read and write the
one palace.

**Why it exists.** The previous adapter filed each memory under the absolute path of a file it also
wrote inside the palace, then wrote the library's bulk miner's private bookkeeping onto every drawer
so that a later ``mempalace mine`` would skip it (issues #741, #751, #831). That is a workaround of a
dependency's internals. Not one of those fields is named in this module, which is how a test proves
none is written (`tests/test_mempalace_provider.py` greps the source for each). The library's own reference
caller, its Hermes provider (``mempalace.integrations.hermes``), files under a name of its own
(``hermes-session:<id>``), writes no file and stamps nothing. This adapter does the same.

**What it does, in the order it does it** (`observe`, the one door every exchange the wake hands
memory goes through):

1. **The door rule for recurring tasks** (``@origin``, 2026-11-23). A task activation is not filed
   when its reply is a near-copy (`_near_copy.near_copy`, 0.8 after ids, times and numbers are
   normalised) of the last *filed* reply for the same task: the same instructions on the same
   timeline (`_exchange_records.task_key`). The first firing is always filed. A near-copy is not
   filed and not recorded. Keyed on the task, never on a word like "check-in".
2. **Already filed?** By name: any drawer filed under this exchange's name, or a drawer the previous
   adapter filed for the same exchange (`_old_drawer`). Then nothing is written: a second filing of
   the same exchange adds nothing.
3. **The record, then the drawers.** The exchange is written to the adapter's own record, outside
   the palace (`_exchange_records`), and then filed through ``file_conversation_exchange``, the
   library's documented write path for live agents, in process. Every drawer is named
   ``basecradle:<timeline uuid>:<item uuid>:<piece number>`` (`_exchange_records.exchange_name`)
   and carries the timeline and the item as two metadata keys of ours (`TIMELINE_KEY`,
   `ITEM_KEY`). Nothing else is written onto a drawer, and no field the library owns is set.

**A reply cut off at its output budget is held at the door** (the capital's decision of 2026-11-25 on
issue #1011, reviewed by ``@briggs``). Its record is written, marked ``cut_off``, and not filed. The
resume that finishes the turn completes the same record (the continuation added after the words so
far) and files it once, whole, with no marker. While the record is held, nothing else files it:
"already filed?" would otherwise find the fragment and skip the whole reply. A wake's catch-up
(`begin_wake`) asks the wake whether the turn can still be resumed (`MemoryProvider.resumable`, read
off the claim on the item); once it cannot (the item stalled, was stopped, dropped, or its timeline
was deleted), the record is filed as it stands, with the one line
`_exchange_records.CUT_OFF_MARKER` after its words. The harness does not finish every resume before
another wake can run catch-up (catch-up runs first in every wake, on every timeline, and a resume
can wait on an outage or a funded account), so the record carries the item's ``kind``, which is what
the question needs.

**The retry, fitted to one process per event** (`_file`). Only the library's own lock refusal
(``mempalace.palace.MineAlreadyRunning``, raised when another process holds the palace's write lock)
is retried, after 1, 2, 4, 8 and 16 seconds. Still busy, the record is left marked unfiled and the
next wake files it (`begin_wake`), after the same "already filed?" check. Any other error is never
retried. Each outcome is one ``memory`` line, by error class.

**What it shares with the previous adapter.** Everything on the read side — the union search, the
``max_distance`` decision, the reranker, the Turn-0 fence and reading guide, the registry-row filter
— is the previous adapter's own, inherited rather than copied, so the two can never search one
palace two ways. What differs is the recall label (`_provenance`), which reads the timeline from a
drawer's own metadata when it carries a BaseCradle name and from the old file name when it does
not, so memories filed by either adapter are labelled correctly.

**Where it differs from the Hermes provider, and why** (the reference for how this library is
called; OpenClaw has no MemPalace provider):

- *The name.* Hermes files every turn of a session under one name, ``hermes-session:<id>``. Here a
  name is unique per exchange and piece: one name per timeline returned 1 of 5 keyword-only memories
  under union search, a unique name 5 of 5 (measured for issue #1011), and a deterministic name is
  what lets a retry find what it already filed.
- *A record of our own.* Hermes keeps none; the drawers are its only copy. Timelines here are deleted
  fast, so the exchange is kept in a format that is ours before it is filed, and a filing the palace
  refused is finished from it.
- *In process, with a bounded retry.* Hermes files from a background thread and drops a turn when
  its queue is full. A wake is one process per event and ends, so there is no thread to outlive it:
  the filing runs in the wake, and the lock refusal is retried and then left to the next wake.
- *The door rule and "already filed?".* Hermes files every turn. The door rule is ``@origin``'s
  ruling on recurring tasks, and the check is what makes a retry, a re-driven turn or a rollback add
  nothing.
- *The text.* Hermes files ``User: … Assistant: …``. This adapter files the exchange format the
  palace already holds (`_mempalace._exchange_markdown`: each incoming line quoted ``> ``), because
  the recall's reading guide tells the model what a quoted line means and every memory filed before
  it is in that format.
- *Split at 800 characters.* Hermes files a turn whole. A long exchange filed whole hid its late
  facts from recall (measured, issue #741), so it is split into the fewest pieces of at most
  `_mempalace.EXCHANGE_PIECE_CHARS` (`_mempalace._pieces`), each its own name.
- *Wing and room.* Hermes classifies the wing by keyword and files in the room ``conversations``.
  Here the wing is the one wing every memory of the agent has always been in (``conversations``) and
  the room is Hermes's, the same for every exchange.
- *Redaction.* The wake blanks every secret before the exchange reaches this adapter (issue #999),
  so the record and the drawers hold the marker, never the secret. If the redaction cannot run,
  nothing reaches the adapter and nothing is written.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from basecradle_harness._exchange_records import (
    NO_TIMELINE,
    Record,
    RecordStore,
    continued,
    exchange_text,
    new_record,
    parse_name,
    records_dir,
    task_key,
)
from basecradle_harness._memory_provider import (
    _MEMPALACE,
    MemoryExchange,
    MemoryScope,
)
from basecradle_harness._mempalace import (
    _A_TIMELINE,
    _ANOTHER_TIMELINE,
    _CONVERSATIONS_WING,
    _FILED,
    _LABEL_SEP,
    _REGISTRY_ROOM,
    _THIS_TIMELINE,
    _TIMELINE_FILE,
    DEFAULT_N_RESULTS,
    MemPalaceMemoryProvider,
    MemPalaceSearchTool,
    _bounded,
    _fenced,
    _filed,
    _import,
    _pieces,
    _source_name,
)
from basecradle_harness._near_copy import near_copy
from basecradle_harness._observability import kv, tool_failed
from basecradle_harness._redaction import scrub
from basecradle_harness._rerank import SURFACE_TOOL, SURFACE_TURN0, MemPalaceReranker
from basecradle_harness._tools import Tool

_log = logging.getLogger("basecradle_harness")

#: The two metadata keys of ours every drawer this adapter files carries, for where-queries: the
#: timeline the exchange happened on and the item (or record) it is named by. Ours, so the library
#: reads neither; no field the library owns is ever written.
TIMELINE_KEY = "basecradle_timeline"
ITEM_KEY = "basecradle_item"

#: The room every drawer is filed in: Hermes's, one for every exchange.
ROOM = "conversations"

#: The seconds slept before each retry of a filing the palace's lock refused. Five retries, 31
#: seconds at most: a wake is one process per event, so past this the next wake files it.
LOCK_RETRY_SECONDS = (1, 2, 4, 8, 16)

# The outcomes of one filing, as the `memory` line's `result` names them (a success names none).
_FILED_OK = "filed"
_BUSY = "busy"
_FAILED = "failed"
_ALREADY = "already_filed"
_NEAR_COPY = "near_copy"
_HELD = "held"  # a reply cut off at its output budget, held unfiled while a resume can still come
_CUT_OFF = "cut_off"
_NO_RESUME = "no_resume"  # a held reply filed with its marker: nothing can resume it any more


class MemPalaceProvider(MemPalaceMemoryProvider):
    """The MemPalace adapter selected by ``mempalace`` (issue #1011). See the module docstring.

    Args:
        palace_path: The palace directory for this agent.
        n_results: As on the previous adapter.
        agent: As on the previous adapter (MemPalace's ``added_by``).
        reranker: As on the previous adapter.
        records: Where the adapter's own records live. Defaults to `records_dir` of the palace: beside
            it, never inside it.
        sleep: How the retry waits. ``time.sleep``; a test passes its own.
    """

    def __init__(
        self,
        palace_path: str | Path,
        *,
        n_results: int = DEFAULT_N_RESULTS,
        agent: str = "harness",
        reranker: MemPalaceReranker | None = None,
        records: str | Path | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        super().__init__(
            palace_path, n_results=n_results, agent=agent, reranker=reranker, name=_MEMPALACE
        )
        self.records = RecordStore(
            records if records is not None else records_dir(self.palace_path)
        )
        self._sleep = sleep

    # --- the door -------------------------------------------------------------------------------

    def observe(self, exchange: MemoryExchange) -> None:
        """The one door every exchange goes through: the door rule, "already filed?", then file.

        The wake has already redacted the exchange (issue #999); an empty one (no user text *and* no
        reply) is skipped. A seeded exchange's names are recorded before it is filed (issue #746), so
        this wake's Turn-0 recall leaves its drawers out however far the filing got.

        **A reply cut off at its output budget is held at the door** (the capital's decision of
        2026-11-25): recorded, ``cut_off``, and not filed. A resume of the same turn completes that
        record and files it once (`_continue`); anything else that reaches a held record adds
        nothing to it. Never a fragment filed beside its whole, and nothing filed is ever replaced.
        """
        if not (exchange.user.strip() or exchange.assistant.strip()):
            return
        record = new_record(
            timeline=exchange.scope.timeline,
            item=exchange.item,
            agent=self.agent,
            user=exchange.user,
            assistant=exchange.assistant,
            seeded=exchange.seeded,
            task_instructions=exchange.task_instructions,
            kind=exchange.kind,
            cut_off=exchange.cut_off,
        )
        held = self._held(record)
        if held is not None:
            self._continue(held, exchange)
            return
        if not record.cut_off and record.task_instructions is not None and self._near_copy(record):
            # A cut-off reply is not judged here: it is not the reply yet. It is judged when it is
            # whole (`_settle`).
            self._log(record, result="skipped", reason=_NEAR_COPY)
            return
        pieces = _pieces(exchange_text(record))
        if exchange.seeded:
            self._seeded_files.update(record.name(n) for n in range(1, len(pieces) + 1))
        try:
            collection = _import("palace").get_collection(str(self.palace_path))
            if self._filed_names(collection, record) or self._old_drawer(
                collection, record, pieces
            ):
                self._log(record, result="skipped", reason=_ALREADY)
                return
        except ImportError:
            raise
        except Exception as error:  # noqa: BLE001 - the record still goes first; the filing reports
            _log.debug("memory already-filed check failed: %s", type(error).__name__)
        self.records.write(record)
        if record.cut_off:
            self._log(record, result=_HELD, reason=_CUT_OFF)
            return
        self._file(record)

    # --- a reply cut off at its output budget ----------------------------------------------------

    def _held(self, record: Record) -> Record | None:
        """The record held for ``record``'s exchange, cut off and not yet filed, or ``None``.

        Read only for an exchange named by an item: an exchange that answered none takes a fresh id
        and can have nothing held. A record that cannot be read is not treated as held, nor as
        absent: it raises, and the wake's guard logs it, because writing over it could lose words.
        """
        held = self.records.read(record.timeline, record.id)
        if held is None or held.filed or not held.cut_off or held.skipped is not None:
            return None
        return held

    def _continue(self, held: Record, exchange: MemoryExchange) -> None:
        """A second exchange of a held turn: a resume completes it; anything else adds nothing.

        A resume's continuation is added after the reply so far (`continued`). Cut off again, the
        record stays held, with every word both attempts wrote. Finished, it is filed once, whole,
        and carries no marker. Anything that is not a resume (the peer's half a stall mines) leaves
        the record as it is: it already holds that half, and whether a resume can still come is the
        claims' to say, at the next wake (`begin_wake`).
        """
        if not exchange.resumed:
            self._log(held, result=_HELD, reason=_CUT_OFF)
            return
        whole = self._across_the_cut(held, exchange.assistant, cut_off=exchange.cut_off)
        if whole is None:
            return
        self.records.write(whole)
        if whole.cut_off:
            self._log(whole, result=_HELD, reason=_CUT_OFF)
            return
        self._settle(whole)

    def _across_the_cut(self, held: Record, continuation: str, *, cut_off: bool) -> Record | None:
        """``held`` completed by ``continuation`` (`continued`), redacted across the cut.

        The wake redacted the fragment and the continuation one at a time (issue #999), so a secret
        the cut split in two is in neither scrub, and joining them would file it whole. So the two
        are scrubbed again as the model wrote them, run together: anything that finds was across the
        cut, and the record then takes that text, the secret blanked and no line break at the seam.
        If the redaction cannot run, nothing is written and the record stays held as it was.
        """
        try:
            run_together = held.assistant + continuation
            seamless = scrub(run_together)
        except Exception:  # noqa: BLE001 - an exchange that cannot be redacted is never filed
            _log.warning(
                "memory %s",
                kv(op="observe", provider=self.name, result="skipped", reason="redaction_failed"),
                exc_info=True,
            )
            return None
        if seamless != run_together:
            return replace(held, assistant=seamless, cut_off=cut_off)
        return continued(held, continuation, cut_off=cut_off)

    def _settle(self, record: Record) -> str:
        """File a record that was held: the door rule first, now that its reply can be judged.

        A completed resume's record carries no marker; one whose resume never came carries
        `CUT_OFF_MARKER` (`exchange_text`). A task reply that turns out to be a near-copy of the
        task's last filed reply is set aside, never filed and never deleted.
        """
        if record.task_instructions is not None and self._near_copy(record):
            self.records.set_aside(record, _NEAR_COPY)
            self._log(record, result="skipped", reason=_NEAR_COPY)
            return _NEAR_COPY
        return self._file(record)

    def _resumable(self, record: Record) -> bool:
        """Whether the turn behind a held ``record`` can still be resumed: the wake's claims say.

        Held while they say it can, while there is no wake to ask (`MemoryProvider.resumable` is
        unset), and while asking fails: a fragment filed before its resume makes "already filed?"
        skip the whole reply, which is the defect this hold exists to prevent, while one held too
        long is still in its record and is filed by a later wake.
        """
        window = self.resumable
        if window is None or record.kind is None or record.timeline == NO_TIMELINE:
            return True
        try:
            return bool(window(record.timeline, record.id, record.kind))
        except Exception as error:  # noqa: BLE001 - cannot tell is "still resumable": hold
            _log.debug("memory resume-window check failed: %s", type(error).__name__)
            return True

    def _near_copy(self, record: Record) -> bool:
        """Whether ``record``'s reply is a near-copy of the last filed reply of the same task."""
        last = self.records.last_task_reply(task_key(record.timeline, record.task_instructions))
        return last is not None and last.filed and near_copy(last.assistant, record.assistant)

    # --- filing ---------------------------------------------------------------------------------

    def _file(self, record: Record, *, check_old: bool = True) -> str:
        """File every piece of ``record`` not yet filed under its name; mark the record filed.

        Retries only the library's lock refusal, after each of `LOCK_RETRY_SECONDS`; still busy, the
        record stays unfiled for the next wake. Any other error is not retried. Returns the outcome
        and writes its one ``memory`` line.
        """
        palace = _import("palace")
        convo_miner = _import("convo_miner")
        busy = getattr(palace, "MineAlreadyRunning", None) or _NeverRaised
        pieces = _pieces(exchange_text(record))
        attempts = 0
        while True:
            attempts += 1
            try:
                collection = palace.get_collection(str(self.palace_path))
                present = self._filed_names(collection, record)
                if check_old and not present and self._old_drawer(collection, record, pieces):
                    self.records.mark_filed(record)
                    self._log(record, result="skipped", reason=_ALREADY, attempts=attempts)
                    return _ALREADY
                for number, piece in enumerate(pieces, start=1):
                    name = record.name(number)
                    if name in present:
                        continue
                    drawer = convo_miner.file_conversation_exchange(
                        collection,
                        wing=_CONVERSATIONS_WING,
                        room=ROOM,
                        text=piece,
                        source_file=name,
                        agent=record.agent,
                        extra_metadata={TIMELINE_KEY: record.timeline, ITEM_KEY: record.id},
                    )
                    if drawer is None:  # a piece is never blank (`_pieces`); upstream changed
                        raise RuntimeError("MemPalace filed no drawer for a non-empty piece")
                    present.add(name)
                break
            except busy:
                if attempts > len(LOCK_RETRY_SECONDS):
                    self._log(record, result=_BUSY, attempts=attempts, pieces=len(pieces))
                    return _BUSY
                self._sleep(LOCK_RETRY_SECONDS[attempts - 1])
            except ImportError:
                raise
            except Exception as error:  # noqa: BLE001 - never retried; the record stays unfiled
                self._log(
                    record,
                    result=_FAILED,
                    error=type(error).__name__,
                    attempts=attempts,
                    pieces=len(pieces),
                    level=logging.WARNING,
                )
                return _FAILED
        filed = self.records.mark_filed(record)
        if filed.task_instructions is not None:
            self.records.remember_task(task_key(filed.timeline, filed.task_instructions), filed)
        self._log(
            filed,
            reason=_NO_RESUME if filed.cut_off else None,
            attempts=attempts,
            pieces=len(pieces),
        )
        return _FILED_OK

    def _filed_names(self, collection, record: Record) -> set[str]:
        """The names already filed for ``record``: every drawer carrying its timeline and item."""
        found = collection.get(
            where={"$and": [{TIMELINE_KEY: record.timeline}, {ITEM_KEY: record.id}]},
            include=["metadatas"],
        )
        names = set()
        for meta in found.metadatas or []:
            name = (meta or {}).get("source_file")
            parsed = parse_name(name)
            if parsed is not None and parsed.id == record.id:
                names.add(name)
        return names

    def _old_drawer(self, collection, record: Record, pieces: list[str]) -> bool:
        """Whether the previous adapter already filed this exchange: a drawer of its first piece.

        The previous adapter named a drawer by the path of a file named for a random uuid, so no
        name can be derived from the exchange to look one up by. What it did write is the same text,
        cut the same way (`_mempalace._pieces`, since 0.164.0), so the first piece is looked for by
        its exact text, in the wing it was filed in, and accepted only from a drawer under an old
        name whose file names this timeline or names none. A backend that cannot answer the query
        answers no: a duplicate is visible and costs a slot; a drop costs a memory.
        """
        if not pieces:
            return False
        first = pieces[0].strip()
        try:
            found = collection.get(
                where={"wing": _CONVERSATIONS_WING},
                where_document={"$contains": first},
                include=["documents", "metadatas"],
            )
        except Exception:  # noqa: BLE001 - see above: cannot tell is "not filed"
            return False
        for document, meta in zip(found.documents or [], found.metadatas or [], strict=False):
            meta = meta or {}
            if (document or "").strip() != first or meta.get("room") == _REGISTRY_ROOM:
                continue
            source = meta.get("source_file")
            if parse_name(source) is not None:
                continue  # a drawer of ours: `_filed_names` is the check for those
            timeline = _timeline_of_old({"source_file": source})
            if timeline is None or timeline == record.timeline:
                return True
        return False

    # --- the next wake ---------------------------------------------------------------------------

    def begin_wake(self) -> None:
        """Forget the previous wake's seeded names, then file every record left unfiled.

        A record is left unfiled when the palace stayed busy past the retry, or the process died
        between writing the record and marking it filed. Each is filed now, after the same "already
        filed?" check, so a piece already in the palace is never filed twice. The palace being busy
        again stops the pass: the rest wait for the next wake rather than each waiting out a retry.
        One ``memory op=refile`` line counts what happened.
        """
        super().begin_wake()
        pending = self.records.unfiled()
        if not pending:
            return
        counts: Counter[str] = Counter()
        for index, (timeline, record_id) in enumerate(pending):
            try:
                record = self.records.resolve_unfiled(timeline, record_id)
            except (ValueError, OSError):
                counts["unreadable"] += 1
                continue
            if record is None or record.filed:
                continue
            if record.skipped is not None:
                # Set aside by the door and killed before its marker went: finish that, never file.
                self.records.set_aside(record, record.skipped)
                counts[_NEAR_COPY] += 1
                continue
            if record.cut_off:
                if self._resumable(record):
                    counts[_HELD] += 1
                    continue
                outcome = self._settle(record)
            else:
                outcome = self._file(record)
            counts[outcome] += 1
            if outcome == _BUSY:
                counts["left"] += len(pending) - index - 1
                break
        _log.info(
            "memory %s",
            kv(
                op="refile",
                provider=self.name,
                records=len(pending),
                filed=counts[_FILED_OK],
                skipped=counts[_ALREADY],
                busy=counts[_BUSY],
                failed=counts[_FAILED],
                unreadable=counts["unreadable"],
                left=counts["left"],
                held=counts[_HELD],
                near_copy=counts[_NEAR_COPY],
            ),
        )

    # --- recall ---------------------------------------------------------------------------------

    def context(self, scope: MemoryScope) -> str | None:
        """The previous adapter's Turn-0 recall, labelled by this adapter's `_provenance`."""
        query = (scope.query or "").strip()
        if not query:
            return None
        hits = self.search(query, surface=SURFACE_TURN0, exclude=frozenset(self._seeded_files))
        if not hits:
            return None
        return _fenced(render_hits(hits, here=scope.timeline))

    def tools(self) -> list[Tool]:
        return [NamedSearchTool(self)]

    # --- the one line per filing ------------------------------------------------------------------

    def _log(
        self,
        record: Record,
        *,
        result: str | None = None,
        reason: str | None = None,
        pieces: int | None = None,
        attempts: int | None = None,
        error: str | None = None,
        level: int = logging.INFO,
    ) -> None:
        """The one ``memory op=observe`` line per exchange the door decided on."""
        timeline = None if record.timeline == NO_TIMELINE else record.timeline
        _log.log(
            level,
            "memory %s",
            kv(
                op="observe",
                provider=self.name,
                result=result,
                reason=reason,
                timeline=timeline,
                pieces=pieces,
                attempts=attempts,
                error=error,
            ),
        )


class NamedSearchTool(MemPalaceSearchTool):
    """The previous adapter's ``memory_search``, labelled by this adapter's `_provenance`."""

    def run(self, query: str, n_results: int | None = None) -> str:
        if not query or not query.strip():
            return tool_failed("Error: 'memory_search' needs a query.", reason="invalid_arguments")
        hits = self.provider.search(
            query.strip(), _bounded(n_results, self.provider.n_results), surface=SURFACE_TOOL
        )
        if not hits:
            return f"No memories match {query!r}."
        here = self._context.timeline if self.bound else None
        return f"Memories matching {query!r}:\n" + render_hits(hits, here=here)


class _NeverRaised(Exception):
    """Stands in for the lock refusal on a MemPalace that defines none, so nothing is retried."""


# --- labels -------------------------------------------------------------------------------------


def render_hits(hits: list[dict], *, here: str | None = None) -> str:
    """`_mempalace._render_hits`, with each chunk led by this adapter's `_provenance`."""
    return "\n".join(f"- {_provenance(hit, here)}{hit['text'].strip()}" for hit in hits)


def _provenance(hit: dict, here: str | None) -> str:
    """The ``(filed … · … timeline …)`` label, for a drawer filed by either adapter.

    The previous adapter's grammar exactly (`_mempalace._provenance`, built from the same words, so
    `_mempalace.QUOTED_LABEL` still recognises it in a reply that quotes it). The timeline is read
    from the drawer's own metadata when it carries a BaseCradle name, and from the old file name
    when it does not (`_timeline_of`). Never from the drawer's text, and never by opening a
    timeline.
    """
    parts = []
    filed = _filed(hit.get("filed_at"))
    if filed:
        parts.append(f"{_FILED}{filed}")
    timeline = _timeline_of(hit)
    if timeline:
        if here is None:
            parts.append(f"{_A_TIMELINE}{timeline}")
        elif timeline == here.lower():
            parts.append(_THIS_TIMELINE)
        else:
            parts.append(f"{_ANOTHER_TIMELINE}{timeline}")
    return f"({_LABEL_SEP.join(parts)}) " if parts else ""


def _timeline_of(hit: dict) -> str | None:
    """The timeline a drawer came from, whichever adapter filed it, or ``None`` when unrecorded."""
    parsed = parse_name(hit.get("source_path") or hit.get("source_file"))
    if parsed is not None:
        return None if parsed.timeline == NO_TIMELINE else parsed.timeline
    return _timeline_of_old(hit)


def _timeline_of_old(hit: dict) -> str | None:
    """The timeline a previous-adapter drawer's file was named for (`_mempalace._TIMELINE_FILE`)."""
    name = _source_name(hit)
    match = _TIMELINE_FILE.match(name) if name else None
    return match["timeline"] if match else None
