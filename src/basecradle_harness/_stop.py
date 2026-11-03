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


def _on_term(signum: int, frame: FrameType | None) -> None:
    """Raise `WakeStopped` once; ignore every later SIGTERM (see the module)."""
    signal.signal(signum, signal.SIG_IGN)
    raise WakeStopped(signum)


def install() -> None:
    """Trap SIGTERM for this process, so a stop unwinds the wake instead of killing it."""
    signal.signal(signal.SIGTERM, _on_term)


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
