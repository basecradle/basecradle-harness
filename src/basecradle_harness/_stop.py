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
Python runs, and a thread that never takes SIGTERM itself reads it and re-signals the main thread
(`signal.pthread_kill`) until the handler has run. A re-signal lands on the blocked main thread,
interrupts its syscall, and the handler runs. In the ordinary case the handler has run before the
relay's first look, and it sends nothing.

A `SIGKILL` runs none of this, and that is the point of the distinction: a crash stays a crash, and
its turn is resumed exactly as before. Only SIGTERM is trapped — SIGINT is Python's own
`KeyboardInterrupt`, which a terminal run already handles, and nothing in the fleet sends another.

Only the `basecradle-harness-wake` entrypoint installs the handler. A library caller's process and
its signals are its own.
"""

from __future__ import annotations

import logging
import os
import signal
import threading
import time
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


#: How long the relay lets the handler run on its own before it re-signals the main thread, and the
#: longest it waits between re-signals after that. The first is far above the microseconds an
#: unblocked main thread takes to run the handler, so an ordinary stop is never re-signalled; the
#: second keeps a stop the main thread keeps missing well inside the router's two seconds.
RELAY_FIRST = 0.02
RELAY_MOST = 0.1

#: Whether `_on_term` has raised this process's stop — set by it, read by the relay, and reset by
#: `install`. A flag as well as `SIG_IGN`, so a re-signal can never raise a second `WakeStopped`.
_stopped = False
#: The write end of the relay's wakeup pipe, once `install` has armed it; `None` before that, and in
#: a forked child, which has no relay thread to read it.
_wakeup: int | None = None


def _on_term(signum: int, frame: FrameType | None) -> None:
    """Raise `WakeStopped` once; ignore every later SIGTERM (see the module)."""
    global _stopped
    signal.signal(signum, signal.SIG_IGN)
    if _stopped:
        return
    _stopped = True
    raise WakeStopped(signum)


def install() -> None:
    """Trap SIGTERM for this process, so a stop unwinds the wake instead of killing it.

    Call it from the main thread, as the entrypoint does: `signal.set_wakeup_fd` refuses any other.
    The wakeup descriptor is process-wide, so this claims it; the entrypoint owns its process's
    signals, and nothing the wake runs sets one of its own (asyncio does, only in a loop that calls
    `add_signal_handler`, which the wake never does).
    """
    global _stopped
    _stopped = False
    # The relay first: its start holds SIGTERM back from this thread for a moment, and a stop held
    # there must meet the default disposition (the wake has not begun), never a handler that would
    # raise out of `install` itself.
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
    """Re-signal `main` after each SIGTERM until `_on_term` has run there (see the module).

    It stops re-signalling the moment the stop is raised, or the moment SIGTERM is no longer ours to
    relay (`_on_term` set `SIG_IGN`, `die` set `SIG_DFL`, or a caller installed a handler of its own),
    so it can never stop a wake that `install` did not arm, and never signal a process that is dying.
    """
    while True:
        try:
            received = os.read(read_end, 512)
        except OSError:
            return
        if not received:
            return
        if signal.SIGTERM not in received:
            continue  # another trapped signal (SIGINT's `KeyboardInterrupt`) is not ours to relay
        wait = RELAY_FIRST
        while not _stopped and signal.getsignal(signal.SIGTERM) is _on_term:
            time.sleep(wait)
            if _stopped or signal.getsignal(signal.SIGTERM) is not _on_term:
                break
            try:
                signal.pthread_kill(main, signal.SIGTERM)
            except (OSError, ValueError):  # the main thread has gone: nothing left to stop
                return
            wait = min(wait * 2, RELAY_MOST)


def _forget_relay_in_child() -> None:
    """A forked child has no relay thread: the parent's pipe must never carry the child's stop.

    Without this, a SIGTERM to the child alone would write to the pipe the parent's relay reads, and
    the parent would stop a wake nobody stopped.
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
