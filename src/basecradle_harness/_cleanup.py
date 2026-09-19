"""Garbage-collect the on-box artifacts of timelines that no longer exist.

When a Timeline is destroyed on the BaseCradle platform, nothing on the fleet
server is cleaned up by itself: the harness persists per-timeline state under
``$HARNESS_HOME`` (chiefly the session transcript, which holds the full
conversation) and has no deletion handler. So a destroyed timeline's content
would survive indefinitely on the box. This module is the periodic **orphan
sweep** that GCs those artifacts — the ``basecradle-harness-cleanup`` entrypoint.

**Sweep-only, by design (settled with the founder).** The platform's
``timeline.deleted`` firehose event is best-effort/droppable, so event-driven
cleanup can never be trusted alone. A periodic sweep is mandatory regardless, and
the *same* sweep cleans up already-deleted timelines for free: the first run on a
box is the backfill — past and future deletions are one identical code path. No
router or Rails change is involved; we are not consuming ``timeline.deleted``.

**The classify switch is the whole feature's safety.** Each referenced UUID is
checked with one cheap ``client.timelines.get(uuid)`` and the *only* outcome that
purges is a clean ``NotFoundError`` (404, confirmed deleted). Every other
outcome — the timeline still exists (200), the agent was merely removed as a
viewer (403), or *any* transient failure (connection, rate-limit, 5xx) — keeps
the artifacts. A platform outage must never be read as "everything deleted" and
trigger a mass purge: we default to **keep** on anything that is not a 404.

**Memory is deliberately out of scope and is never touched.** The sweep operates
only on the six artifact dirs below; it never enumerates, and so never deletes,
``memory.db`` (+ ``-wal``/``-shm``) or the MemPalace palace dir. If a peer told
the agent its birthday on a since-deleted timeline, the agent must still remember
it. See the CLAUDE.md "Security invariants" section.

The six artifact kinds, all under ``$HARNESS_HOME``, keyed by timeline UUID with
the same ``quote(..., safe='')`` filename convention the stores already use:

================  ==========================================================
Kind              Path
================  ==========================================================
Session           ``sessions/timeline%3A<uuid>.json``  (source ``timeline:<uuid>``)
High-water marks  ``marks/<uuid>.txt``, ``marks/<kind>/<uuid>.txt``
Seen-set (tasks)  ``seen/<kind>/<uuid>.txt``
Claims            ``claims/<kind>/<uuid>/*.claim``  (per-uuid directory)
Wake-breaker      ``breaker/<uuid>.wakes``, ``breaker/<uuid>.tripped``
Billing-blocked   ``billing/<uuid>.blocked``  (out-of-funds debounce, issue #336)
================  ==========================================================

**Stranded temps are swept on every run, whatever their timeline** (issue #526). Each atomic
write stages a temp, and a writer killed inside the write window leaves it behind with nothing to
remove it: a copy of a conversation beside a live transcript, a copy of a live token beside
``agent.env``. `prune_stranded_temps` removes those — by exact name, in the exact places the
harness stages writes, and only once `STRANDED_AFTER` has passed and any pid the writer stamped is
dead — so no live writer's temp is ever touched. It reaches outside ``$HARNESS_HOME`` in two places
and only by name: the env file's directory and the top level of ``~/.mempalace`` (never the palace
beneath it).

The sweep is idempotent and crash-safe: a re-run re-derives the artifact set from
disk, and a half-done purge finishes on the next run. There is no concurrency
hazard — a 404 timeline is terminal, so no live wake for it can be in flight (a
wake on a deleted timeline already errors in ``WakeAgent.__init__``). And it makes
**no provider/LLM call anywhere** (the "zero token burn at rest" fleet rule); the
only cost is one ``timelines.get`` per referenced UUID.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

from basecradle._exceptions import BaseCradleError, ForbiddenError, NotFoundError

from basecradle_harness._basecradle import _client_from_env, _configure_logging
from basecradle_harness._install import config_home
from basecradle_harness._mempalace import _CLI_CONFIG_DIR, _CLI_CONFIG_FILE
from basecradle_harness._observability import RED, head, kv
from basecradle_harness._token import TEMP_PREFIX, TEMP_SUFFIX
from basecradle_harness._version import __version__
from basecradle_harness._wake import _process_alive

_log = logging.getLogger("basecradle_harness")

#: The session ``source`` a wake runs under (``WakeAgent.source``), so a timeline's
#: transcript file is ``sessions/{quote("timeline:<uuid>")}.json``. Only sessions whose
#: decoded source carries this prefix are timeline artifacts — a ``github:`` session is a
#: different channel and is never swept.
_TIMELINE_SOURCE_PREFIX = "timeline:"

#: A transcript staged by `Session._save`'s atomic write and left behind by a process killed
#: mid-save: ``sessions/<quoted-source>.json.<pid>-<token>.tmp`` (issue #297).
#:
#: It is swept because of what it *contains*. On success the temp is renamed into place and on an
#: exception it is removed, so one only exists when a wake was killed inside the write window — and
#: it then holds **the entire conversation**. A sweep that purged `…json` and walked past
#: `…json.4213-9f2c.tmp` would report a deleted timeline as purged while leaving its transcript on
#: the box forever, which is precisely the outcome this module exists to prevent.
_SESSION_TEMP = re.compile(r"^(?P<source>.+)\.json\.(?P<stamp>[^.]+)\.tmp$")

#: A mark staged by `MarkStore.set` and left behind by a process killed mid-write:
#: ``marks/<quoted-timeline>.txt.<pid>.tmp`` (messages) or ``marks/<kind>/…`` (issue #526). It is
#: purged with its timeline, and swept as a stranded temp on a live one.
_MARK_TEMP = re.compile(r"^(?P<timeline>.+)\.txt\.(?P<pid>\d+)\.tmp$")

#: How long a staged temp must sit untouched before the sweep calls it **stranded** (issue #526).
#:
#: Every temp below lives for one atomic write: stage, (fsync), rename or link, and a handler removes
#: it on an exception. Only a writer killed *inside* that window — `SIGKILL`, the OOM killer, a power
#: loss, where no handler runs — leaves one behind, and then nothing ever removes it: the orphan
#: sweep reaches a session temp only once its timeline is deleted, and nothing reaches the rest at
#: all. A write takes milliseconds, so an hour is far past any live one. Where the writer stamps its
#: pid, the pid must also be dead: a temp is removed only when **both** say its writer is gone, so a
#: live writer's temp is never touched even if the two clocks or pid namespaces disagree.
STRANDED_AFTER = 60 * 60

#: ``claims/<kind>/<timeline>/<uuid>.claim.<pid>.tmp`` — `ClaimStore._write`, staging a commit,
#: an abandon, or the claim a take-over writes.
_CLAIM_WRITE_TEMP = re.compile(r"^.+\.claim\.(?P<pid>\d+)\.tmp$")

#: ``claims/<kind>/<timeline>/.<uuid>.<wake>.new`` and ``….takeover.new`` — the populated records
#: `ClaimStore.claim` and `ClaimStore.reclaim` link into place. A take-over *token*
#: (``.<uuid>.takeover.<owner>``) never ends in ``.new``, so it can never match.
_CLAIM_LINK_TEMP = re.compile(r"^\..+\.new$")

#: ``.basecradle-env.<random>.tmp`` beside the env file — `_token._atomic_write`. It holds a copy of
#: the whole env file, **live ``BASECRADLE_TOKEN`` included** (mode 600).
_ENV_TEMP = re.compile(rf"^{re.escape(TEMP_PREFIX)}.+{re.escape(TEMP_SUFFIX)}$")

#: ``~/.mempalace/.config.json.<hex>.tmp`` — `_mempalace._write_cli_config`. It holds the MemPalace
#: CLI config, which can carry an embeddings API key.
_MEMPALACE_TEMP = re.compile(rf"^\.{re.escape(_CLI_CONFIG_FILE)}\.[0-9a-f]+\.tmp$")


@dataclass
class SweepSummary:
    """The one-line outcome of a sweep: how each referenced timeline was classified.

    ``checked`` is every distinct UUID enumerated from the artifact dirs;
    ``purged`` + ``kept`` + ``kept_forbidden`` + ``skipped_transient`` partition it.
    """

    checked: int = 0
    purged: int = 0
    kept: int = 0
    kept_forbidden: int = 0
    skipped_transient: int = 0
    #: Stranded temps removed this run, on any timeline or none (`prune_stranded_temps`).
    stranded_temps: int = 0

    def __str__(self) -> str:
        return (
            f"cleanup sweep: checked {self.checked} timeline(s) — "
            f"purged {self.purged}, kept {self.kept}, "
            f"kept-forbidden {self.kept_forbidden}, skipped-transient {self.skipped_transient}; "
            f"removed {self.stranded_temps} stranded temp(s)"
        )


def enumerate_artifacts(home: Path) -> dict[str, list[Path]]:
    """Map each referenced timeline UUID to the on-disk paths that belong to it.

    Scans the six artifact dirs under ``home``, parsing each timeline UUID out of a
    filename/dirname and URL-decoding it (``unquote``), the exact inverse of the
    ``quote(..., safe='')`` the stores write with — so encode/decode round-trips. The
    returned paths are what a purge deletes, so this is the single source of truth for
    *what exists* (re-deriving it each run is what makes the sweep idempotent).

    A path is a plain file for every kind except claims, where the per-UUID *directory*
    ``claims/<kind>/<uuid>/`` (holding its items' ``.claim`` records) is the unit to remove.
    """
    artifacts: dict[str, list[Path]] = {}

    def add(uuid: str, path: Path) -> None:
        artifacts.setdefault(uuid, []).append(path)

    # Session transcripts — `sessions/{quote(source)}.json`. Only timeline-sourced
    # sessions are ours; a `github:`/other channel session decodes without the prefix
    # and is left strictly alone (it has nothing to do with a deleted timeline).
    #
    # …and any transcript *staged* by the atomic save and orphaned by a killed wake
    # (`<name>.json.<pid>-<token>.tmp`, see `_SESSION_TEMP`). It holds the whole
    # conversation, so purging the transcript while leaving the temp would purge nothing.
    sessions = home / "sessions"
    if sessions.is_dir():
        for path in sessions.iterdir():
            if path.name.endswith(".json"):
                encoded = path.name[: -len(".json")]
            elif staged := _SESSION_TEMP.match(path.name):
                encoded = staged.group("source")
            else:
                continue
            source = unquote(encoded)
            if source.startswith(_TIMELINE_SOURCE_PREFIX):
                add(source[len(_TIMELINE_SOURCE_PREFIX) :], path)

    # Marks — `marks/<uuid>.txt` (messages, the original flat layout) and
    # `marks/<kind>/<uuid>.txt` (assets, webhook_events). `rglob` catches both depths.
    marks = home / "marks"
    if marks.is_dir():
        for path in marks.rglob("*.txt"):
            add(unquote(path.stem), path)
        for path in marks.rglob("*.tmp"):  # …and a mark a killed write staged (`_MARK_TEMP`)
            if staged := _MARK_TEMP.match(path.name):
                add(unquote(staged.group("timeline")), path)

    # Seen-sets — `seen/<kind>/<uuid>.txt` (tasks today; any future kind for free).
    seen = home / "seen"
    if seen.is_dir():
        for path in seen.rglob("*.txt"):
            add(unquote(path.stem), path)

    # Claims — a per-uuid directory `claims/<kind>/<uuid>/` of the timeline's `.claim` records.
    # The directory is the unit to purge, so track it (not its individual files).
    claims = home / "claims"
    if claims.is_dir():
        for kind_dir in claims.iterdir():
            if not kind_dir.is_dir():
                continue
            for uuid_dir in kind_dir.iterdir():
                if uuid_dir.is_dir():
                    add(unquote(uuid_dir.name), uuid_dir)

    # Wake-breaker — `breaker/<uuid>.wakes` and `breaker/<uuid>.tripped`.
    breaker = home / "breaker"
    if breaker.is_dir():
        for path in breaker.glob("*"):
            if path.suffix in (".wakes", ".tripped"):
                add(unquote(path.stem), path)

    # Billing-blocked markers — `billing/<uuid>.blocked` (issue #336). A stale marker for a deleted
    # timeline is inert (that timeline never re-wakes), but it lives beside the other per-timeline
    # stores and is purged with them so a deleted timeline leaves nothing behind on the box.
    billing = home / "billing"
    if billing.is_dir():
        for path in billing.glob("*.blocked"):
            add(unquote(path.stem), path)

    return artifacts


def purge(paths: list[Path]) -> None:
    """Delete the artifact paths for one timeline — files unlinked, dirs removed wholesale.

    Tolerant of a missing path (a concurrent or prior partial purge) **and of a per-path
    failure**: a single un-deletable artifact (a permission error, a TOCTOU vanish) is logged
    and stepped over, never raised — so one bad file can't abort the sweep and strand every
    other orphan timeline this run. Files and dirs are handled symmetrically here (both
    error-tolerant), which is also what makes ``sweep`` resilient and the re-run idempotent.
    Only ever called with paths that ``enumerate_artifacts`` produced, so it can never reach
    ``memory.db`` or the palace.
    """
    for path in paths:
        try:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                # `missing_ok` closes the exists()->unlink() TOCTOU (a concurrent --timeline
                # purge, a crash-recovery overlap) without a raise.
                path.unlink(missing_ok=True)
        except OSError as error:
            _log.warning("cleanup: could not remove %s: %s", path, error)


def prune_stranded_temps(home: Path, *, now: float | None = None) -> list[Path]:
    """Remove every temp a killed write left behind, on live timelines too (issue #526).

    The orphan sweep removes a *session* temp only once its timeline is deleted, and nothing else
    ever removed a stranded temp at all. So a wake killed mid-save left a full copy of the
    conversation beside the live transcript for the life of the timeline, and a killed token
    refresh left a copy of a live ``BASECRADLE_TOKEN`` beside ``agent.env`` forever. This pass looks
    in exactly the places the harness stages writes, for exactly the names it stages them under:

    ==============================================  ===================================  =======
    Temp                                            Writer                               Pid
    ==============================================  ===================================  =======
    ``sessions/<source>.json.<pid>-<token>.tmp``    `Session._save`                      yes
    ``marks/[<kind>/]<timeline>.txt.<pid>.tmp``     `MarkStore.set`                      yes
    ``claims/…/<uuid>.claim.<pid>.tmp``             `ClaimStore._write`                  yes
    ``claims/…/.<uuid>.<wake>[.takeover].new``      `ClaimStore.claim` / ``reclaim``     no
    ``.basecradle-env.<random>.tmp``                `_token._atomic_write`               no
    ``~/.mempalace/.config.json.<hex>.tmp``         `_mempalace._write_cli_config`       no
    ==============================================  ===================================  =======

    A temp is removed when it is older than `STRANDED_AFTER` **and**, where it carries one, its pid
    is not a live process. The env temp is looked for beside ``BASECRADLE_ENV_FILE`` and in the
    config home (where ``agent.env`` lives); the MemPalace one only at the top level of
    ``~/.mempalace``, which is never walked, so the palace beneath it is never reached. Nothing that
    does not match one of those names is ever touched, and a temp that cannot be read or removed is
    logged and left for the next run.
    """
    now = time.time() if now is None else now
    removed: list[Path] = []

    def consider(path: Path, pid: int | None) -> None:
        try:
            age = now - path.stat().st_mtime
        except OSError:
            return  # gone already, or unreadable: nothing to decide
        if age < STRANDED_AFTER or (pid is not None and _writer_alive(pid)):
            return  # young enough to be mid-write, or its writer is still alive
        try:
            path.unlink(missing_ok=True)
        except OSError as error:
            _log.warning("cleanup: could not remove stranded temp %s: %s", path, error)
            return
        removed.append(path)
        _log.info("cleanup: removed stranded temp %s (untouched %.0fs)", path, age)

    for path in _entries(home / "sessions"):
        if staged := _SESSION_TEMP.match(path.name):
            pid, _, _ = staged.group("stamp").partition("-")
            consider(path, int(pid) if pid.isdigit() else None)

    for folder in [home / "marks", *_entries(home / "marks")]:
        for path in _entries(folder):
            if staged := _MARK_TEMP.match(path.name):
                consider(path, int(staged.group("pid")))

    for kind in _entries(home / "claims"):
        for folder in _entries(kind):
            for path in _entries(folder):
                if written := _CLAIM_WRITE_TEMP.match(path.name):
                    consider(path, int(written.group("pid")))
                elif _CLAIM_LINK_TEMP.match(path.name):
                    consider(path, None)

    env_dirs = {config_home()}
    if env_file := os.environ.get("BASECRADLE_ENV_FILE"):
        env_dirs.add(Path(env_file).expanduser().parent)
    for folder in sorted(env_dirs):
        for path in _entries(folder):
            if _ENV_TEMP.match(path.name):
                consider(path, None)

    for path in _entries(Path.home() / _CLI_CONFIG_DIR):
        if _MEMPALACE_TEMP.match(path.name):
            consider(path, None)

    return removed


def _entries(folder: Path) -> list[Path]:
    """The entries of `folder`, or none — a missing or unreadable directory never stops the pass."""
    try:
        return list(folder.iterdir()) if folder.is_dir() else []
    except OSError as error:
        _log.warning("cleanup: could not list %s: %s", folder, error)
        return []


def _writer_alive(pid: int) -> bool:
    """Is the writer that stamped `pid` still running? A number no process can hold is not alive."""
    try:
        return _process_alive(pid)
    except (OverflowError, ValueError):
        return False  # `os.kill` rejects it outright: nothing by that pid can be writing


def classify(client: object, uuid: str) -> str:
    """Classify one timeline by a single ``timeline.get`` — the safety switch of the sweep.

    Returns one of ``"purge"`` / ``"keep"`` / ``"keep_forbidden"`` / ``"skip_transient"``.
    Only a clean 404 (``NotFoundError``) purges; **everything else keeps**, because the one
    failure mode that must never happen is reading a platform outage as a mass deletion.

    The except order matters: ``NotFoundError`` and ``ForbiddenError`` are siblings (both
    ``BaseCradleError``), and ``NotAViewerError`` is a ``ForbiddenError`` — so 404 first,
    403 next, then a broad catch-all that buckets every transient/unexpected error as
    *keep-and-retry-next-run*. ``BaseException`` (``KeyboardInterrupt``) is intentionally
    *not* caught, so an operator can still abort the sweep.
    """
    try:
        client.timelines.get(uuid)  # type: ignore[attr-defined]
    except NotFoundError:
        return "purge"
    except ForbiddenError:
        # The timeline exists; the agent was merely removed as a viewer. Keep its artifacts
        # (out of scope here — a possible follow-up), and log it so it is visible.
        return "keep_forbidden"
    except Exception:  # noqa: BLE001 — deliberately broad: any non-404 defaults to keep.
        # Connection / rate-limit / 5xx / generic BaseCradleError, or any unexpected error:
        # transient. Skip this UUID this run and retry next sweep. Never purge on doubt.
        return "skip_transient"
    return "keep"


def sweep(home: Path, client: object) -> SweepSummary:
    """Enumerate every referenced timeline, classify it, and purge only the confirmed-deleted.

    The whole GC: derive the artifact set from disk, ask the platform about each UUID once,
    and act on *only* a clean 404. Makes no provider/LLM call — the sole cost is one cheap
    ``timelines.get`` per referenced timeline.
    """
    artifacts = enumerate_artifacts(home)
    summary = SweepSummary()
    for uuid in sorted(artifacts):
        summary.checked += 1
        verdict = classify(client, uuid)
        if verdict == "purge":
            purge(artifacts[uuid])
            summary.purged += 1
            _log.info("cleanup: purged artifacts for deleted timeline %s", uuid)
        elif verdict == "keep_forbidden":
            summary.kept_forbidden += 1
            _log.info("cleanup: kept timeline %s (403 — exists, agent not a viewer)", uuid)
        elif verdict == "skip_transient":
            summary.skipped_transient += 1
            _log.warning("cleanup: skipped timeline %s this run (transient error)", uuid)
        else:
            summary.kept += 1
    summary.stranded_temps = len(prune_stranded_temps(home))
    return summary


def purge_one(home: Path, uuid: str) -> list[Path]:
    """Unconditionally purge a single timeline's artifacts — the manual ``--timeline`` ops path.

    No ``timelines.get`` and no classify: the operator has asserted this timeline is gone, so
    its enumerated artifacts are removed outright. Returns the paths purged (empty if the box
    held nothing for it). The encode-decode round-trip means the operator passes a *plain*
    UUID and it still matches the percent-encoded on-disk names.
    """
    paths = enumerate_artifacts(home).get(uuid, [])
    purge(paths)
    return paths


def _resolve_home() -> Path:
    """``HARNESS_HOME`` or a clear error — the same var the wake persists every artifact under."""
    home = os.environ.get("HARNESS_HOME")
    if not home:
        raise ValueError(
            "Cleanup requires HARNESS_HOME — the directory where the agent's per-timeline "
            "artifacts (sessions, marks, seen, claims, breaker) persist."
        )
    return Path(home)


def main(argv: list[str] | None = None) -> int:
    """The ``basecradle-harness-cleanup`` entrypoint: GC deleted timelines' on-box artifacts.

    ``--sweep`` is the scheduled GC (and the first-run backfill); ``--timeline <uuid>`` is a
    manual one-off purge for ops. Exit 0 on success; non-zero on a hard config/auth failure.
    """
    parser = argparse.ArgumentParser(
        prog="basecradle-harness-cleanup",
        description=(
            "Garbage-collect the on-box artifacts of deleted BaseCradle timelines under "
            "HARNESS_HOME. Memory (memory.db + the MemPalace palace) is never touched."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"basecradle-harness-cleanup {__version__}",
        help="print the installed basecradle-harness version, then exit.",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help=(
            "enumerate every referenced timeline, classify each via one timelines.get, and "
            "purge only those the platform 404s (confirmed deleted). The first run backfills "
            "already-deleted timelines. No model call."
        ),
    )
    parser.add_argument(
        "--timeline",
        metavar="UUID",
        help=(
            "manually purge a single timeline's artifacts unconditionally (no platform check) "
            "— an ops escape hatch. The scheduled path uses --sweep."
        ),
    )
    args = parser.parse_args(argv)

    if not args.sweep and not args.timeline:
        parser.error("one of --sweep or --timeline <uuid> is required")

    # Configure a stderr handler (systemd captures it) so the INFO summary is visible,
    # unless an embedding app already configured logging. Shared with the wake CLI, so both
    # default to INFO and honor HARNESS_LOG_LEVEL (raising it past INFO quiets the summary).
    _configure_logging()

    try:
        home = _resolve_home()
        if args.timeline:
            purged = purge_one(home, args.timeline)
            _log.info(
                "cleanup: manually purged %d artifact path(s) for timeline %s",
                len(purged),
                args.timeline,
            )
        else:
            summary = sweep(home, _client_from_env())
            _log.info("%s", summary)
    except (BaseCradleError, ValueError, KeyError) as error:
        # A hard setup failure — no/expired credentials (`_client_from_env` or a mint),
        # an unreadable home — surfaces as a clean non-zero exit with a one-line message
        # for the NOC's journal, never a raw traceback. (Per-UUID platform errors during
        # the sweep are already absorbed by `classify` as transient-keep.)
        #
        # It goes through the logger as an ERROR as well as to stderr, for the same reason the
        # wake CLI's does (issue #272): a bare print is unleveled, so the sweep that never ran is
        # invisible to a severity filter — and a silently-failing sweep looks exactly like a
        # sweep with nothing to do.
        _log.error("%s %s", head("cleanup failed", RED), kv(error=str(error)))
        print(f"basecradle-harness-cleanup: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
