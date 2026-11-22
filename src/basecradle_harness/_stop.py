"""A stopped wake is a stop: SIGTERM ends the wake as `stopped`, and no later wake resumes it.

Issue #798. The router's sanctioned `stop-wake` op (basecradle-router#386) SIGTERMs every process in
a wake — the harness included — and `runuser` SIGKILLs it about two seconds later. Before this
module the harness installed no handler, so `SIGTERM`'s default disposition killed it exactly as a
crash does: no `finally` ran, the transcript ended on a tool call with no result, the claim stayed
`in-flight`, and the next organic wake did what the delivery guarantee tells it to do with a dead
wake's turn — it **resumed** it (`resuming … reason=died_mid_tool_chain`), re-issuing its creates
and handing the model the turn to finish. A human stopped the wake, and the harness undid the stop.

The two seconds are enough to say what happened, and that is all they are used for:

- `install` turns the first SIGTERM into a `WakeStopped` raised in the main thread, so the stack
  unwinds through every `finally` the wake already has. A second SIGTERM is **ignored**: the router
  signals every process in the wake, `sudo` relays what it receives to its child, so one stop can
  arrive twice, and a second raise would land inside the first one's bookkeeping.
- `Session` gives each call the stop left unanswered the `STOPPED` result (never `INTERRUPTED`, the
  sentinel the recovery re-issues creates from) and ends the turn on `STOPPED_NOTE`.
- `WakeAgent` settles every claim the wake still holds in flight as stopped and writes
  ``wake end … outcome=stopped signal=TERM``.
- `die` then ends the process **by the signal** — the default disposition restored, the signal
  re-sent to itself — so `sudo` still dies by -15 and the router's `stage=wake outcome=stopped`
  classification holds. An exit code that merely looked like 143 would not be the same thing.

**A stop that has arrived is acted on even when the main thread is already blocked** (issue #990).
Python runs the handler only on the main thread, and only between bytecodes or when a blocking
syscall the main thread is in returns `EINTR`. A SIGTERM the main thread took on its way *toward* a
read — or one the kernel handed to another thread — trips the handler's flag and interrupts nothing,
so the handler waits behind the read for as long as the read takes: a model call's whole fitted
timeout, which the router's SIGKILL two seconds later turns into a crash. CI caught it about one
stop in ten landing as a model call began. So `install` also arms a **relay**: CPython writes the
signal's number to a wakeup descriptor (`signal.set_wakeup_fd`) from its C-level handler, before any
Python runs, and a thread that never takes SIGTERM itself reads it and wakes the main thread
(`signal.pthread_kill` with a no-op `WAKE_SIGNAL`, never SIGTERM) until the stop has been raised. The
wake interrupts the blocked syscall, Python runs every tripped handler, and the stop's is among them
only if a stop really arrived, so the relay can never stop a wake by itself. In the ordinary case the
handler has run before the relay's first look, and it sends nothing.

A block a stop must not cut in half is held by the stop, never by a signal mask (`holding`): a mask
only moves SIGTERM to whichever thread has it unblocked, Python runs the handler on the main thread
regardless, and the relay's wake would interrupt the block anyway.

A `SIGKILL` runs none of this, and that is the point of the distinction: a crash stays a crash, and
its turn is resumed exactly as before. Only SIGTERM is trapped — SIGINT is Python's own
`KeyboardInterrupt`, which a terminal run already handles, and nothing in the fleet sends another.

Only the `basecradle-harness-wake` entrypoint installs the handler. A library caller's process and
its signals are its own.
"""

from __future__ import annotations

import contextlib
import logging
import os
import signal
import threading
import time
from collections.abc import Iterator
from types import FrameType
from typing import NoReturn


class WakeStopped(BaseException):
    """The wake received SIGTERM and is stopping (issue #798).

    A `BaseException`, never an `Exception`, for the same reason `KeyboardInterrupt` is one: every
    `except Exception` the wake has — a tool's failure fed back to the model, a per-item degrade, a
    best-effort write — must let a stop through rather than treat it as one more fault to survive.
    """

    def __init__(self, signum: int = signal.SIGTERM) -> None:
        self.signum = signum
        super().__init__(signal.Signals(signum).name)

    @property
    def signal_name(self) -> str:
        """The signal's name without its ``SIG`` prefix — ``TERM`` — as the log line carries it."""
        return signal.Signals(self.signum).name.removeprefix("SIG")


#: The signal the relay wakes the main thread with. **Never SIGTERM**: re-sending the stop would let
#: the relay *manufacture* one from a byte that was not this process's (a forked child's C handler
#: writes to the same pipe), and could land inside `_on_term` before it sets `SIG_IGN`. Waking with a
#: no-op signal instead interrupts the blocked syscall, and Python then runs every handler whose flag
#: is tripped — SIGTERM's among them only if a stop has really arrived. SIGWINCH is harmless when
#: anything else sends it (a terminal resize) and nothing in the fleet sends it on purpose.
WAKE_SIGNAL = signal.SIGWINCH
#: How long the relay lets the handler run on its own before it wakes the main thread, and the
#: longest it waits between wakes after that. The first is far above the microseconds an unblocked
#: main thread takes to run the handler, so an ordinary stop is never relayed; the second keeps a
#: stop the main thread keeps missing well inside the router's two seconds.
RELAY_FIRST = 0.02
RELAY_MOST = 0.1
#: How long the relay goes on waking the main thread after a SIGTERM's byte: well past the router's
#: SIGKILL. A bound by time, never by evidence: whether a wake found a stop is not something the
#: relay can read without a race, and a byte that was no stop (a forked child's) then costs a few
#: harmless wakes and nothing more. A stop still pending past it runs at the main thread's next
#: bytecode, exactly as before the relay.
RELAY_FOR = 5.0

#: Whether `_on_term` has taken this process's stop — set by it, read by the relay, and reset by
#: `install`. A flag as well as `SIG_IGN`, so a stop is raised once whatever else re-enters it.
_stopped = False
#: How many `holding` blocks the main thread is inside, and the stop one of them held back.
_holds = 0
_held_back: int | None = None
#: The write end of the relay's wakeup pipe, once `install` has armed it; `None` before that, and in
#: a forked child, which has no relay thread to read it.
_wakeup: int | None = None


def _on_term(signum: int, frame: FrameType | None) -> None:
    """Raise `WakeStopped` once — or hold it for `holding` — and ignore every later SIGTERM."""
    global _stopped, _held_back
    signal.signal(signum, signal.SIG_IGN)
    if _stopped:
        return
    _stopped = True
    if _holds:
        _held_back = signum
        return
    raise WakeStopped(signum)


def _on_wake(signum: int, frame: FrameType | None) -> None:
    """The relay's wake does nothing: interrupting the main thread's syscall is the whole of it."""


@contextlib.contextmanager
def holding() -> Iterator[None]:
    """Hold a stop back from the block, and raise it once the block is done.

    For a block a stop must not cut in half (`_journal.owned_streams` restoring descriptors 1 and 2
    and draining their readers). It is the stop that is held, not the signal: a thread's signal mask
    only moves SIGTERM to whichever thread has it unblocked, and Python then runs the handler on the
    main thread whatever the main thread's mask says. A no-op off the main thread, where no stop is
    ever raised.
    """
    global _holds, _held_back
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    _holds += 1
    try:
        yield
    finally:
        _holds -= 1
        if not _holds and _held_back is not None:
            signum, _held_back = _held_back, None
            raise WakeStopped(signum)


def install() -> None:
    """Trap SIGTERM for this process, so a stop unwinds the wake instead of killing it.

    Call it from the main thread, as the entrypoint does: `signal.signal` refuses any other, and it
    is called first, so a misplaced call raises before a relay is aimed at the wrong thread. The
    wakeup descriptor is process-wide, so this claims it; the entrypoint owns its process's signals,
    and nothing the wake runs sets one of its own (asyncio does, only in a loop that calls
    `add_signal_handler`, which the wake never does).
    """
    global _stopped
    _stopped = False
    signal.signal(WAKE_SIGNAL, _on_wake)
    # The relay before the SIGTERM handler: its start holds SIGTERM back from this thread for a
    # moment, and a stop held there must meet the default disposition (the wake has not begun),
    # never a handler that would raise out of `install` itself.
    _arm_relay()
    signal.signal(signal.SIGTERM, _on_term)


def _arm_relay() -> None:
    """Start the relay once per process and point the wakeup descriptor at it."""
    global _wakeup
    if _wakeup is None:
        read_end, write_end = os.pipe()
        os.set_blocking(write_end, False)  # `set_wakeup_fd` requires it: the C handler cannot wait
        # Born with SIGTERM blocked, so the relay can never be the thread the stop is delivered to:
        # a thread inherits its creator's mask, and this one must hold it from its first instruction.
        signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
        try:
            threading.Thread(
                target=_relay,
                args=(read_end, threading.get_ident()),
                name="harness-stop-relay",
                daemon=True,
            ).start()
        finally:
            signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM})
        _wakeup = write_end
    signal.set_wakeup_fd(_wakeup, warn_on_full_buffer=False)


def _relay(read_end: int, main: int) -> None:
    """After each SIGTERM, wake `main` until its stop has been taken there (see the module).

    It stops waking the main thread the moment the stop is taken, the moment SIGTERM is no longer
    ours to relay (`_on_term` set `SIG_IGN`, `die` set `SIG_DFL`, a caller installed a handler of its
    own), or after `RELAY_FOR`. It sends only `WAKE_SIGNAL`, so it can never stop a wake itself.
    """
    while True:
        try:
            received = os.read(read_end, 512)
        except OSError:
            return
        if not received:
            return
        if signal.SIGTERM not in received:
            continue  # its own wake, or SIGINT's `KeyboardInterrupt`: not a stop to relay
        until = time.monotonic() + RELAY_FOR
        wait = RELAY_FIRST
        while True:
            time.sleep(wait)
            if _stopped or signal.getsignal(signal.SIGTERM) is not _on_term:
                break
            if time.monotonic() >= until:
                break
            try:
                signal.pthread_kill(main, WAKE_SIGNAL)
            except (OSError, ValueError):  # the main thread has gone: nothing left to stop
                return
            wait = min(wait * 2, RELAY_MOST)


def _forget_relay_in_child() -> None:
    """A forked child has no relay thread, so it must stop writing to the parent's pipe.

    Its C handler would otherwise write the child's own signals there, and wake the parent's relay
    for a stop that is not the parent's (harmless wakes, since the relay never sends a stop, but the
    child's signals are not the parent's business). Runs for `os.fork`, `multiprocessing` and `preexec_fn`;
    `subprocess`'s own fork-then-exec runs no Python in the child, and its signals are its own after
    the exec.
    """
    global _wakeup
    if _wakeup is not None:
        try:
            signal.set_wakeup_fd(-1)
        except ValueError:  # not the main thread of the child: it has no Python wakeup to clear
            pass
        os.close(_wakeup)
        _wakeup = None


os.register_at_fork(after_in_child=_forget_relay_in_child)


def die(stop: WakeStopped) -> NoReturn:
    """End the process by `stop`'s signal, after flushing the log.

    Dying by a signal skips `atexit` and every buffered handler, so the log is shut down first: the
    `wake end … outcome=stopped` line is the one record of what the stop did, and it must not be
    left in a buffer. The `os._exit` after the kill is unreachable while the default disposition is
    terminate; it exists so this function can never return into code that believes the wake ended.
    """
    logging.shutdown()
    signal.signal(stop.signum, signal.SIG_DFL)
    os.kill(os.getpid(), stop.signum)
    os._exit(128 + stop.signum)  # pragma: no cover - the kill above has already ended the process
