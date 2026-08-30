"""``basecradle-harness-scrub-palace`` — delete harness scaffolding from a mined palace.

The backward-facing half of the mining boundary (issue #438). `_mining` closes the paths by
which the harness's own composed text could reach a mining memory provider; this removes what
already got in, from a palace that was mined before the fix.

**Dry run is the default, and it is also the discovery pass.** With no ``--apply`` nothing is
deleted: every matching chunk is printed in full, grouped by the catalog class that matched,
so a human can read what is about to go. A second list — ``REVIEW`` — carries chunks that
contain scaffolding *beside real content*; those are never deleted by any invocation, and they
are how a class of scaffolding nobody has enumerated yet becomes visible. Anything
scaffolding-shaped in that list goes to review and, if confirmed, into `_mining.catalog()` —
then the dry run is repeated. It never becomes an improvised delete.

**What makes deleting safe is `_mining.classify`, not this file.** A chunk qualifies only when
*every* content-bearing line in it is an exact catalog literal, so a real memory that merely
mentions memory, tools, or the brief cannot match, and one that quotes a scaffolding line and
then says something real is downgraded to review. There is no similarity, no threshold, and no
fuzz anywhere in the path.

**The source file goes with the drawers, and that is not tidiness — it is the whole reason a
scrub holds.** The adapter mines *files*: `observe` writes each exchange under
``<palace>/conversations/`` and re-mines the directory, and MemPalace decides what to skip by
asking whether the file's drawers are still in the palace and *complete*
(``palace.file_already_mined`` / ``prefetch_mined_set``, which omit a source whose surviving
drawers are short of their recorded ``chunk_total``). Delete a drawer and leave its file, and
the very next wake re-mines that file and puts the scaffolding straight back. So a file that
lost a drawer is removed with it — only ever inside the palace's own ``conversations``
directory, never a path outside it.

**Run ``--apply`` with the agent's wakes held.** The dry run is a pure read and is safe at any
time; an apply deletes drawers and their mining input while a concurrent wake could be mining new
ones into the same palace. The harness takes no lock of its own here on purpose — the wake-lock
belongs to whatever is launching wakes (the NOC on the fleet), and a second, private lock that
only this command respects would read as a guarantee it cannot give.

Nothing here writes to the palace on a dry run, and nothing here ever touches a drawer the
report did not name.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from basecradle_harness._memory_provider import _palace_path
from basecradle_harness._mempalace import _CONVERSATIONS_WING, _import
from basecradle_harness._mining import Verdict, catalog, classify
from basecradle_harness._observability import GREEN, RED, YELLOW, head, kv
from basecradle_harness._version import __version__

_log = logging.getLogger("basecradle_harness")

#: How many drawers one `collection.get` page reads. MemPalace's own bulk scans use 1000; a
#: palace is tens of thousands of drawers, and the whole point of paging is to never hold it all.
_PAGE = 1000

#: How many ids one `collection.delete` call carries. Deleting in pages for the same reason
#: reading does, and small enough that a backend with a statement-parameter ceiling is not the
#: thing that decides whether a scrub completes.
_DELETE_BATCH = 200


@dataclass
class Finding:
    """One classified chunk: what it is, where it came from, and what it says."""

    drawer_id: str
    verdict: Verdict
    classes: tuple[str, ...]
    text: str
    source_file: str | None


@dataclass
class Report:
    """What one pass over a palace found — the dry run's whole output, and `apply`'s input."""

    palace: Path
    total: int = 0
    scrub: list[Finding] = field(default_factory=list)
    review: list[Finding] = field(default_factory=list)

    @property
    def kept(self) -> int:
        """Chunks no catalog class matched at all — the ordinary memories."""
        return self.total - len(self.scrub) - len(self.review)


def scan(collection, palace: Path) -> Report:
    """Classify every drawer in the palace, paging so a large one is never held whole.

    Reads documents and metadata only — no embeddings, no query, no write. `source_file` is
    read off each drawer's metadata because it is what `apply` needs to keep the scrub from
    being undone by the next wake's re-mine (see the module docstring).
    """
    report = Report(palace=palace)
    offset = 0
    while True:
        page = collection.get(limit=_PAGE, offset=offset, include=["documents", "metadatas"])
        ids = list(page.get("ids") or [])
        if not ids:
            break
        documents = list(page.get("documents") or [])
        metadatas = list(page.get("metadatas") or [])
        for index, drawer_id in enumerate(ids):
            report.total += 1
            text = documents[index] if index < len(documents) else None
            if not text:
                continue
            verdict, classes = classify(text)
            if verdict is Verdict.KEEP:
                continue
            meta = (metadatas[index] if index < len(metadatas) else None) or {}
            finding = Finding(
                drawer_id=drawer_id,
                verdict=verdict,
                classes=classes,
                text=text,
                source_file=meta.get("source_file"),
            )
            (report.scrub if verdict is Verdict.SCRUB else report.review).append(finding)
        offset += len(ids)
    return report


def apply(collection, report: Report) -> tuple[int, int]:
    """Delete exactly the drawers the report put in `scrub`, and the convo files they came from.

    Returns ``(drawers deleted, source files removed)``. Nothing in `review` is touched, on any
    invocation — a chunk holding real content is a memory whatever else is stapled to it.

    A source file is removed only when it lies inside this palace's own ``conversations``
    directory. The check is a resolved-path containment test, not a string prefix: the path came
    out of a metadata field, and a scrub must not be able to delete something elsewhere on the
    box because a drawer's provenance said so.

    **Files first, drawers second, and the order is the crash behavior.** A scrub that dies
    part-way through leaves the palace in one of two states, and only one of them is safe: unlink
    first, and the worst case is scaffolding drawers still in the palace with no file left to
    re-mine them from — exactly the status quo, which the next run finishes. Delete first, and the
    worst case is drawers gone with their files still on disk, which the very next wake re-mines,
    putting the scaffolding back. A file left orphaned of its drawers costs nothing (nothing reads
    it but the miner, which now skips it); a drawer left orphaned of its file costs a re-mine.
    """
    removed = 0
    for path in _convo_files(report):
        try:
            path.unlink()
            removed += 1
        except FileNotFoundError:
            pass  # already gone: another pass, or an operator's own cleanup. Not a failure.
        except OSError as error:
            # Loud, and not fatal: the drawers still go, and a file the next wake re-mines is a
            # re-pollution the next scrub catches. Silence here would leave it invisible.
            _log.warning("scrub %s", kv(op="unlink", path=str(path), error=str(error)))
    ids = [finding.drawer_id for finding in report.scrub]
    for start in range(0, len(ids), _DELETE_BATCH):
        collection.delete(ids=ids[start : start + _DELETE_BATCH])
    return len(ids), removed


def _convo_files(report: Report) -> list[Path]:
    """The mining input files behind the scrubbed drawers, restricted to this palace's own dir."""
    root = (report.palace / _CONVERSATIONS_WING).resolve()
    files: dict[Path, None] = {}
    for finding in report.scrub:
        if not finding.source_file:
            continue
        path = Path(finding.source_file)
        try:
            resolved = path.resolve()
        except OSError:  # pragma: no cover - a path that cannot even be resolved is not ours
            continue
        if resolved == root or root not in resolved.parents:
            continue
        files[resolved] = None
    return list(files)


# --- rendering ---------------------------------------------------------------


def render(report: Report, *, applied: bool) -> str:
    """The report a human reads before approving an apply — every match, in full.

    Full text and not an excerpt, deliberately: this is the artifact a capital review gates the
    apply on, and an elided chunk is one nobody can actually judge. The catalog's `why` for each
    class is printed once, at the top, so a reviewer does not have to open this repo to know what
    each finding is claiming to be.
    """
    reasons = {entry.name: entry.why for entry in catalog()}
    lines = [f"palace: {report.palace}", f"drawers: {report.total}", ""]
    lines += _section(
        "SCRUB",
        (
            "harness scaffolding — every line matched the catalog"
            + (". DELETED." if applied else ". Deleted by --apply.")
        ),
        report.scrub,
        reasons,
    )
    lines += _section(
        "REVIEW",
        "scaffolding beside real content — never deleted; read these and extend the catalog",
        report.review,
        reasons,
    )
    lines.append(
        "summary: "
        + kv(
            scrub=len(report.scrub),
            review=len(report.review),
            keep=report.kept,
            total=report.total,
            mode="apply" if applied else "dry-run",
        )
    )
    if not applied:
        lines.append("Nothing was deleted. Re-run with --apply to delete the SCRUB list.")
    return "\n".join(lines)


def _section(
    title: str, subtitle: str, findings: list[Finding], reasons: dict[str, str]
) -> list[str]:
    """One titled block of findings, or a one-line "none" when there are none."""
    lines = [f"{title} — {len(findings)} chunk(s): {subtitle}"]
    if not findings:
        return [*lines, "  (none)", ""]
    for name in sorted({name for finding in findings for name in finding.classes}):
        lines.append(f"  · {name}: {reasons.get(name, '')}")
    lines.append("")
    for finding in findings:
        lines.append(f"  [{'+'.join(finding.classes)}] {finding.drawer_id}")
        if finding.source_file:
            lines.append(f"    source: {finding.source_file}")
        lines += [f"    | {line}" for line in finding.text.splitlines() or [""]]
        lines.append("")
    return lines


# --- the CLI -----------------------------------------------------------------


def open_palace(palace: Path):
    """The palace's drawer collection, or a `ValueError` naming what is wrong.

    ``create=False``, always: this command exists to read and prune an existing palace, and a
    typo in ``--palace`` must not scaffold an empty one and report a clean bill of health on it.

    Every way MemPalace can refuse to open one becomes a `ValueError` carrying the vendor's own
    exception type and message. It is caught broadly on purpose: the refusals are backend-specific
    classes (`CollectionNotInitializedError`, an embedder-identity mismatch, a backend the palace
    was not written with) living inside an *optional extra* this module cannot import to name, and
    the operator running this on a box needs the wrong-path case to read as one line, not as a
    traceback ending in a class they have never heard of. Nothing is swallowed — the vendor's text
    rides inside, which is the same relay-the-verdict-verbatim stance the harness takes with a
    provider error.
    """
    if not palace.is_dir():
        raise ValueError(f"no palace at {palace} — pass --palace, or set MEMPALACE_PALACE_PATH")
    try:
        return _import("palace").get_collection(str(palace), create=False)
    except ImportError:
        raise  # the extra is missing: `_import`'s own message already says how to fix it
    except Exception as error:  # a vendor refusal of any class — relayed, never swallowed
        raise ValueError(
            f"could not open the palace at {palace} — {type(error).__name__}: {error}. "
            "Is this the agent's palace directory, and was it ever mined?"
        ) from error


def resolve_palace(explicit: str | None) -> Path:
    """``--palace`` → the same resolution a wake uses (`MEMPALACE_PALACE_PATH`, `$HARNESS_HOME`).

    Deliberately the adapter's own function and not a second copy of its rules: a scrub that
    resolved the palace differently from the agent would faithfully report on the wrong mind.
    """
    if explicit:
        return Path(os.path.abspath(os.path.expanduser(explicit)))
    home = os.environ.get("HARNESS_HOME")
    return _palace_path(home)


def main(argv: list[str] | None = None) -> int:
    """The ``basecradle-harness-scrub-palace`` entrypoint. Exit 0 on success, 1 on a bad palace."""
    parser = argparse.ArgumentParser(
        prog="basecradle-harness-scrub-palace",
        description=(
            "Find (and with --apply, delete) harness-composed scaffolding mined into a MemPalace "
            "palace before the mining boundary was enforced. Dry run by default."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"basecradle-harness-scrub-palace {__version__}",
        help="print the installed basecradle-harness version, then exit.",
    )
    parser.add_argument(
        "--palace",
        metavar="DIR",
        help=(
            "the palace directory to operate on. Defaults to the one this agent would bind: "
            "MEMPALACE_PALACE_PATH, else $HARNESS_HOME/mempalace."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "delete exactly the chunks the dry run reports under SCRUB, and the conversation "
            "files they were mined from (so the next wake cannot re-mine them). REVIEW findings "
            "are never deleted. Run this with the agent's wakes held."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    try:
        palace = resolve_palace(args.palace)
        collection = open_palace(palace)
        report = scan(collection, palace)
        deleted = removed = 0
        if args.apply and report.scrub:
            deleted, removed = apply(collection, report)
        print(render(report, applied=args.apply))
        colour = RED if report.scrub else GREEN
        _log.info(
            "%s %s",
            head("palace scrub", colour if args.apply else YELLOW),
            kv(
                palace=str(palace),
                mode="apply" if args.apply else "dry-run",
                drawers=report.total,
                scrub=len(report.scrub),
                review=len(report.review),
                deleted=deleted if args.apply else None,
                files_removed=removed if args.apply else None,
                after=report.total - deleted,
            ),
        )
    except (ValueError, ImportError) as error:
        _log.error("%s %s", head("palace scrub failed", RED), kv(error=str(error)))
        print(f"basecradle-harness-scrub-palace: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
