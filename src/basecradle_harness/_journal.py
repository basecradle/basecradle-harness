"""The wake owns its stdout and stderr: every byte written to them reaches the journal as whole records.

A deployed wake's stdout and stderr are **one** journal stream: the router's wake-runner wraps the
harness in ``systemd-cat``, which connects both descriptors to the same journald socket, and journald
cuts that stream into records at each newline. So whatever writes to either descriptor decides where
a record starts, and the fleet recognises a contract line by its head at a record's start
(basecradle-noc#1518). Two failures follow, and both were found in the fleet's stored lines
(basecradle#868, issue #912):

- **an unfinished line glues onto the next record.** A write that ends without a newline — a
  MemPalace mining progress line, a block-buffered ``print`` flushed at its buffer's edge, a child
  process's last word — leaves the stream mid-line, and the harness's next record is appended to it.
  Three real ``wake end`` lines were stored that way, their head mid-record, and are counted by no
  column.
- **a newline in text the harness did not write starts a record that text controls.** Python prints
  a traceback over many lines, and a library or a child writes what it likes.

The harness's own records are already one line each (`_observability.ScrubbingFormatter`). This
module closes the rest at the descriptor, where no writer can go around it: for the life of the wake
(`owned_streams`) descriptors 1 and 2 are pipes the harness reads, and each line any writer sends —
this process's ``print``, a C library's ``fprintf``, a subprocess that inherited them — is logged as
a record of its own, ``INFO stdout: "…"`` or ``WARNING stderr: "…"``, the line flattened, scrubbed
and quoted. An unfinished line is held until its newline arrives, or until the wake ends, and is
then a whole record; it can never be the start of someone else's. The harness's handler writes to a
duplicate of the original stderr, so its own records are never read back as output.

Three things are stated rather than hidden:

- **Order across the two paths is not exact.** A pipe is read on its own thread, so an output line
  can land a few milliseconds after a harness record written later.
- **Output after the wake is raw.** `owned_streams` hands descriptors 1 and 2 back when it exits, so
  a write during interpreter shutdown goes out as written. It follows ``wake end``, so the only
  record it could be glued to is a crash report written after it. The two shutdown writers the harness knows of are routed
  through the handler instead (`route_crashes`): Python's report of an exception nothing caught, in
  the main thread or another, and of one raised where Python can only ignore it. A
  ``KeyboardInterrupt`` keeps Python's own report, which only a person at a terminal sees.
- **A descendant that outlives the wake** keeps the pipe open, so `owned_streams` waits a bounded
  time for the readers (`DRAIN_SECONDS`, in all) and then lets the process end without them; what
  the descendant writes later is lost rather than glued.
- **Only stderr is checked.** Descriptor 1 is taken whatever it is, so a wake run with stdout sent
  to a file sends it to the journal instead. A wake's stdout carries nothing of its own.

An output line is logged at its descriptor's level, ``INFO`` or ``WARNING``, so ``HARNESS_LOG_LEVEL``
governs it as it governs every other record. The handler's report of its own failure is one line on
the journal (`_observability.JournalHandler`), never a write to ``sys.stderr``: that is the pipe, and
the reader waiting on the handler's lock would never drain it.

Installed only by the wake entrypoint, only when it configured logging itself
(`_basecradle._configure_logging`), and only while ``sys.stderr`` is descriptor 2: an embedding
application, a test runner, and a caller that redirected ``sys.stderr`` keep their own descriptors.
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import signal
import sys
import threading
import time
from collections.abc import Iterator
from types import TracebackType
from typing import Any

from basecradle_harness._attempts import another_try_follows
from basecradle_harness._observability import Rendered, _full_value

_log = logging.getLogger("basecradle_harness")

#: The longest unfinished line held before it is logged anyway. A progress bar redrawn with ``\r``
#: and never ended would otherwise grow without bound; past this it is logged as a whole record
#: (the formatter splits any record over `_observability.RECORD_BYTES`), and reading carries on.
PENDING_BYTES = 64 * 1024

#: How long the wake's end waits, in all, for the readers to log what their pipes still hold. Ample
#: for a pipe's whole buffer; a descendant holding a pipe open makes the wait the whole of it, never
#: longer, and short enough that a stopped wake still dies by its signal before ``runuser`` kills it.
DRAIN_SECONDS = 1.0

#: The record name every output line is logged under — the harness's, so the formatter treats the
#: record as its own, its quoted line a `Rendered` argument.
_LOGGER = "basecradle_harness.output"
_output_log = logging.getLogger("basecradle_harness.output")


@contextlib.contextmanager
def owned_streams(handler: logging.StreamHandler) -> Iterator[None]:
    """For the life of the block, descriptors 1 and 2 are pipes read into whole records via `handler`.

    `handler` is the one the entrypoint installed; it is pointed at a duplicate of the original
    stderr, so it keeps writing to the journal while descriptor 2 is a pipe, and keeps that stream
    after the block. On exit Python's buffers are flushed into the pipes, the original descriptors
    are restored (which ends each pipe for its reader), and each reader logs what it still holds.

    A no-op unless `handler` writes to ``sys.stderr`` and ``sys.stderr`` is descriptor 2: a caller
    that redirected either one has said where the output goes, and it is not the descriptors.
    """
    if handler.stream is not sys.stderr or not _is_descriptor(sys.stderr, 2):
        yield
        return
    for stream in (sys.stdout, sys.stderr):
        _flush(stream)
    journal = os.fdopen(
        os.dup(2),
        "w",
        encoding=getattr(sys.stderr, "encoding", None) or "utf-8",
        errors="backslashreplace",
        buffering=1,
    )
    handler.setStream(journal)
    readers = []
    saved = []
    try:
        for fd, name, level in ((1, "stdout", logging.INFO), (2, "stderr", logging.WARNING)):
            read_end, write_end = os.pipe()
            saved.append((fd, os.dup(fd)))
            os.dup2(write_end, fd)
            os.close(write_end)
            reader = threading.Thread(
                target=_pump,
                args=(read_end, name, level, handler),
                name=f"harness-{name}",
                daemon=True,
            )
            reader.start()
            readers.append(reader)
        yield
    finally:
        # A SIGTERM landing here would raise out of a half-restored pair, so it waits until both
        # descriptors are back and the readers have drained; it is then raised as the wake's stop.
        blocked = _block_term()
        try:
            for stream in (sys.stdout, sys.stderr):
                _flush(stream)
            for fd, original in saved:
                os.dup2(original, fd)
                os.close(original)
            deadline = time.monotonic() + DRAIN_SECONDS
            for reader in readers:
                reader.join(max(0.0, deadline - time.monotonic()))
        finally:
            if blocked:
                signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM})


def _block_term() -> bool:
    """Hold SIGTERM back from this thread, when this is the thread that takes it."""
    if threading.current_thread() is not threading.main_thread():
        return False
    signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
    return True


def _is_descriptor(stream: Any, fd: int) -> bool:
    try:
        return stream.fileno() == fd
    except (AttributeError, OSError, ValueError):  # io.UnsupportedOperation is both of the last two
        return False


def _flush(stream: Any) -> None:
    with contextlib.suppress(Exception):  # a closed or detached stream has nothing to flush
        stream.flush()


def _pump(fd: int, name: str, level: int, handler: logging.Handler) -> None:
    """Read `fd` to its end, logging each line through `handler` as one record.

    Logged straight to `handler`, never through a logger: a handler some library attached to the
    root that writes to ``sys.stderr`` would otherwise send each line back into this pipe, forever.
    """
    pending = b""
    with io.FileIO(fd, "r") as pipe:
        while chunk := pipe.read(65536):
            *lines, pending = (pending + chunk).split(b"\n")
            for line in lines:
                _emit(line, name, level, handler)
            if len(pending) > PENDING_BYTES:
                _emit(pending, name, level, handler)
                pending = b""
    _emit(pending, name, level, handler)


def _emit(line: bytes, name: str, level: int, handler: logging.Handler) -> None:
    text = line.decode("utf-8", errors="replace")
    if not text.strip() or not _output_log.isEnabledFor(level):
        return
    record = logging.LogRecord(
        _LOGGER, level, __file__, 0, f"{name}: %s", (Rendered(_full_value(text)),), None
    )
    handler.handle(record)


def route_crashes() -> None:
    """Route Python's own reports of an exception nothing caught through the logging handler.

    Each would otherwise print a multi-line traceback straight to stderr — after the wake, where
    `owned_streams` no longer owns it, in the case that matters most: the main thread's, which Python
    reports once `main` has already returned. Through the handler it is one record, its traceback a
    quoted string at the line's end. A ``KeyboardInterrupt`` keeps Python's own report.
    """
    sys.excepthook = _uncaught
    threading.excepthook = _uncaught_in_thread
    sys.unraisablehook = _unraisable


def _uncaught(kind: type[BaseException], error: BaseException, trace: TracebackType | None) -> None:
    if issubclass(kind, KeyboardInterrupt):
        sys.__excepthook__(kind, error, trace)
        return
    # Python exits `1` on it, and the router runs an exit `1` again while a quick try remains (issue
    # #962), so until the last try it is not yet a failure: the report keeps its traceback, at
    # WARNING. A thread's report stays ERROR, because a thread dying does not end the wake.
    level = logging.WARNING if another_try_follows() else logging.ERROR
    _log.log(level, "The wake raised an exception nothing caught.", exc_info=(kind, error, trace))


def _uncaught_in_thread(args: threading.ExceptHookArgs) -> None:
    if args.exc_type is SystemExit:
        return  # what Python's own hook does: a thread that exits says nothing
    thread = args.thread.name if args.thread is not None else "unknown"
    _log.error(
        "Thread %s raised an exception nothing caught.",
        thread,
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
    )


def _unraisable(args: Any) -> None:
    error = args.exc_value
    _log.warning(
        "Python ignored an exception it could not raise (%s).",
        args.err_msg or "Exception ignored in",
        exc_info=(args.exc_type, error, args.exc_traceback) if error is not None else None,
    )
