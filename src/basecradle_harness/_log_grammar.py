"""Prove the *billing* log lines still exist — the log-grammar probe (basecradle-noc#509).

The fleet's **LLM Vendor Payment Failed** alert (basecradle-noc#317) is founder-named and pages a
human: an agent's model account runs out of prepaid credit, the agent goes silent, and only a
person adding funds can fix it. It fires off one derived column, ``billing_blocked``, whose whole
definition is a regex over two lines *this package writes* — the onset
``wake reported_failure … kind=billing`` and the debounced repeat ``wake billing_blocked``
(`_report.billing_onset_line` / `_report.billing_repeat_line`).

**Both of those lines exist only on the failure path**, so on a healthy fleet nothing in the
pattern ever arrives. That is not a small gap. The NOC's extraction guard
(``fleetops/extraction_drift.py``) watches every derived column by asking whether it is still
extracting anything off live traffic — and a column nothing exercises cannot be watched that way
at all. It had already happened once: the fleet-wide colour roll (basecradle-harness#414) repainted
both heads and broke **both** of this column's clauses at once, caught only because two sibling
builders read their emitting side before shipping. A rename would take the founder's page silently
dark, and nothing on any dashboard would move.

The NOC cannot close that from its side — it cannot make a vendor account run out of money, and it
would not want a monitor that could. **The property belongs to the emitter, so the check does
too.** This module is that check.

What it does, and the division of labour
----------------------------------------

Called as ``basecradle-harness-log-grammar billing_blocked`` — by the NOC's ``run-claim-probe``
enumerated op, resolved from this agent's root-owned claims manifest and exec'd as the agent's own
OS user — it renders both clauses **through the very functions the real failure path renders them
with**, writes them to journald, and reads them back to prove they landed. From there the fleet's
ordinary shipping path (journald → Vector → Better Stack) carries them, and the extraction guard
reads them off the same live stream it already reads and asserts the column extracted.

Neither repo ever holds the other's artifact, which is the whole point:

- the **harness** proves *"I still render these bytes"* — this probe's exit code, in the claims
  ledger;
- the **NOC** proves *"I still extract them"* — its guard's verdict, off the live stream.

The harness never sees the regex; the NOC never sees the emitter. A probe that tried to assert
extraction would need the ClickHouse tables, which is exactly the second spelling of another
repo's contract this design exists to avoid.

Why it does not page the founder
--------------------------------

A synthetic line that reads as a real billing failure to the alarm *is* a real page. The
distinguisher is not new — it is the wake-origin contract (basecradle-noc#473, @origin
2026-08-11), generalized by the capital on 2026-08-18: **``source=probe`` marks a line
manufactured by the fleet's own instrumentation, never real traffic.** The NOC already carries a
``source`` label column and a byte-identical block-list predicate
(``coalesce(label('source'), '') != 'probe'``) on four production charts; *LLM Vendor Payment
Failures* gains the same one.

``billing_blocked``'s own expression is **untouched**, and that is load-bearing rather than tidy:
filtering the probe out inside the column would make the guard prove an expression production
traffic never hits — the gap re-opened one level down. And the filter direction is @origin's
ratified block-list, not an allow-list: if the ``source`` label ever stops extracting, these lines
**flood** the alarm rather than a real out-of-funds event being silently dropped. Loud beats quiet.

Three consequences that look like details and are not
-----------------------------------------------------

- **The stamp is not a parameter.** `lines` always passes ``source=PROBE_SOURCE``; there is no
  "quiet" mode to get wrong. A probe line that lost its stamp is a page to a human's phone.
- **The synthetic carries no ``provider=``, and the reason is a *neighbouring* instrument.** The
  real billing lines carry it, but ``provider`` is the declared witness **parent** for two other
  columns (``llm_missing_tokens``, ``tool_cost``, both parented on ``' llm provider='``). A probe
  line carrying ``provider=`` would let that parent reach its line gate in hours when no real
  ``llm`` line arrived at all — at which point the guard asks whether ``' llm provider='``
  extracted anything, gets no, and calls two healthy columns deaf. A monitor that manufactures
  false positives in an instrument beside it is worse than the gap it closes. The general rule,
  for anything added here later: **carry no field that is a witness parent for another column**,
  and keep every value a bare token (`kv` quotes a value holding a space, but the fleet's label
  extractors are naive regexes over the whole message, so a ``reason="… provider=x …"`` would
  populate ``provider`` from *inside* the quotes).
- **Both clauses are emitted, separately.** The guard asks only whether a column extracted
  *anything*, and ``billing_blocked`` is two OR'd clauses — so one working clause would green a
  column whose other clause has gone deaf. They are different heads, in different branches,
  painted different colours; a future rename will very plausibly hit one and not both. Emitting
  each as its own line is what lets the NOC judge them independently.

Why its own journald identifier
-------------------------------

Not ``basecradle-wake-<slug>``, which is where the real lines live. Three reasons, and the third
is the one with teeth: that identifier is the router wake-runner's contract and spelling it here
would be the transcription this fleet forbids twice; an agent's wake journal is the flight
recorder a human digs into on the rare day something breaks, and salting it with synthetic
failures makes that record actively misleading; and the fleet's ``error_lines`` column is
identifier-scoped to ``basecradle-router`` and ``basecradle-wake-*``, so a wake identifier would
contaminate *Server Errors* — one of only two charts the fleet's alarm spec records as
deliberately carrying no filter. Under its own identifier this contributes nothing to it at any
severity, and a human reading a Live Tail sees ``[basecradle-log-grammar]`` and knows at a glance
that the line is instrumentation.

The line is written at **INFO** and wearing the same `LOG_FORMAT` envelope every harness line
wears, so the fleet's ``level`` column reads it honestly rather than seeing a severity-less line.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time

from basecradle_harness._observability import LOG_FORMAT, agent_slug
from basecradle_harness._report import (
    PROBE_SOURCE,
    billing_onset_line,
    billing_repeat_line,
)

#: The journald identifier these synthetic lines are written under — this repo's to own, per the
#: capital's ruling 6 (2026-08-18): *a probe line lands under an identifier its emitting repo owns,
#: and that identifier is what the NOC's witness `parent` declares.* See the module docstring for
#: why it is deliberately not the wake identifier.
IDENTIFIER = "basecradle-log-grammar"

#: The NOC derived columns this probe can exercise, keyed by the column's **exact** spelling. The
#: claim id is ``log-grammar:<column>`` (the capital's ruling 5): the claims ledger is the one
#: sanctioned meeting point between an emitter and the monitor, and a claim id is shared vocabulary
#: by design — the same way ``overlay-tool:shell`` is.
BILLING_BLOCKED = "billing_blocked"
COLUMNS = (BILLING_BLOCKED,)

#: The console script's own name, so the claims emitter and this module cannot disagree about what
#: the manifest's ``cmd`` should invoke.
SCRIPT = "basecradle-harness-log-grammar"

#: Age-of-proof threshold for the claim (`class: rare` — silence is the normal state, so the proof
#: is a forced exercise). One hour, and it is doing two jobs honestly: the evidence should never be
#: older than the hourly drift pass that reports it, and the NOC's extraction guard needs these
#: lines arriving often enough to judge inside a window. The exerciser's own arithmetic holds at
#: this value (``ttl - CLAIM_REFRESH_MARGIN + period`` = 1 − 0.75 + 0.5 = 0.75 h < 1 h). The NOC
#: owns the guard's constants and may re-set this number in its wiring phase; a fire costs two log
#: lines — no model call, no vendor credit, no platform I/O.
TTL_HOURS = 1

#: The ``reason=`` slug the synthetic lines carry. Deliberately *not* the production slug
#: (``out_of_funds``): ``reason`` populates no derived column, so it is free to say what is true,
#: and a human in a Live Tail should need no lookup to know what they are looking at. A bare token
#: by construction — see the module docstring on quoted values.
REASON = "log_grammar_probe"

#: ``EX_TEMPFAIL``. The claims contract's one load-bearing exit code: *we ran and could not
#: determine the answer*, recorded as ERROR → ``unprovable``. Any **other** non-zero lands as
#: FAIL — *we asked; the capability is broken* — which would misdescribe a run that established
#: nothing. Distinct is not softer: both are red.
EX_TEMPFAIL = 75

#: How long to wait for journald to make a just-written line readable. Ingestion is asynchronous,
#: so the read-back polls rather than reads once. Generous on purpose: the failing direction here
#: is a FAIL recorded against a healthy box.
_READBACK_TIMEOUT_S = 10.0
_READBACK_INTERVAL_S = 0.25

#: How far *before* the write the read-back looks — and it must stay small, which is the opposite
#: of the instinct. Every fire emits **byte-identical** lines (that is the whole point: they are
#: the production grammar), so a wide window can match a *previous* fire's lines and report
#: "landed" about a write that never did. Widening this "to be safe" is precisely the change that
#: silently downgrades the claim from *in the journal* back to *rendered*. One second only exists
#: to absorb the sub-second gap between `time.time()` here and journald's own receive stamp, on
#: the same clock; the exerciser's period is half an hour, so nothing legitimate is within reach.
_READBACK_LOOKBACK_S = 1

#: journald's native datagram socket — what ``systemd-cat`` itself writes to. Used directly so the
#: probe depends on the journal being there rather than on a binary being on a ``PATH`` that
#: ``env -i`` may not carry.
_JOURNAL_SOCKET = "/run/systemd/journal/socket"


class Unprovable(Exception):
    """We could not ask. Raised where the honest answer is `EX_TEMPFAIL`, never a FAIL."""


def lines(column: str, *, agent: str) -> list[str]:
    """The synthetic lines for `column` — one per clause, rendered by the production renderers.

    Every line is stamped ``source=probe`` unconditionally: the stamp is what keeps a synthetic out
    of the founder's page, so it is not a parameter a caller can omit.
    """
    if column != BILLING_BLOCKED:  # pragma: no cover - `main` validates before reaching here
        raise Unprovable(f"unknown column: {column}")
    return [
        billing_onset_line(reason=REASON, source=PROBE_SOURCE, agent=agent),
        billing_repeat_line(reason=REASON, source=PROBE_SOURCE, agent=agent),
    ]


def record(line: str, *, level: str = "INFO") -> str:
    """One line as it reaches the journal: the `LOG_FORMAT` envelope around the rendered bytes.

    The severity token is part of the *envelope*, not of the grammar under proof — but it has to be
    there, because nothing on an agent box sets a syslog priority and the fleet's ``level`` column
    reads severity out of the message text or not at all.
    """
    return LOG_FORMAT % {"levelname": level, "message": line}


def _field(key: str, value: str) -> bytes:
    """One journald native-protocol field.

    Two encodings, and picking the wrong one is a **forgery vector** rather than a formatting bug.
    The short ``KEY=value`` form is terminated by a newline, so a value that itself contains one
    makes journald read the remainder as *more fields* — a message could append its own
    ``PRIORITY=`` or ``SYSLOG_IDENTIFIER=`` and land wherever it liked. The protocol's answer is a
    binary form (``KEY``, newline, a little-endian 64-bit length, the raw bytes, newline), used
    here whenever a value carries a newline.

    Nothing this module writes today can: `kv` flattens every value to one line before it renders,
    precisely so a record is never split into unleveled fragments. This is written for the change
    that has not happened yet — the failure would be silent, and it would be in the one direction
    that matters (a synthetic reaching a page-the-human alert wearing a forged identity).
    """
    if "\n" not in value:
        return f"{key}={value}\n".encode()
    raw = value.encode()
    return key.encode() + b"\n" + len(raw).to_bytes(8, "little") + raw + b"\n"


def _journal_datagram(message: str, *, priority: int = 6) -> bytes:
    """One journald native-protocol datagram: the three fields a `systemd-cat` line carries."""
    fields = {
        "MESSAGE": message,
        "PRIORITY": str(priority),
        "SYSLOG_IDENTIFIER": IDENTIFIER,
    }
    return b"".join(_field(key, value) for key, value in fields.items())


def emit(messages: list[str]) -> None:
    """Write `messages` to journald under `IDENTIFIER`.

    Raises `Unprovable` when the journal is not reachable — a box with no journald is one where we
    never got to ask the question, not one where the answer is no.
    """
    if not os.path.exists(_JOURNAL_SOCKET):
        raise Unprovable(f"no journald socket at {_JOURNAL_SOCKET}")
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
            for message in messages:
                sock.sendto(_journal_datagram(message), _JOURNAL_SOCKET)
    except OSError as error:
        raise Unprovable(f"could not write to journald: {error}") from error


def _read_back(messages: list[str], *, since: float) -> bool:
    """Whether every message in `messages` is readable back out of the journal.

    The claim this upgrades is *"rendered"* → *"in the journal"*, which is the difference between a
    write call returning and a line actually existing for Vector to ship. Verified live on the AI
    box (2026-08-18): an agent's own OS user reads its own entries back with no group membership.

    Polls until `_READBACK_TIMEOUT_S`, because journald ingestion is asynchronous. A `journalctl`
    that cannot run at all raises `Unprovable`; one that runs and keeps not finding the lines is a
    genuine FAIL. The window is deliberately narrow — see `_READBACK_LOOKBACK_S`.
    """
    # `--since` reads **local** time (journalctl's own default), so the stamp is built in local
    # time on purpose; rendering it as UTC would silently shift the window by the box's offset.
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(since - _READBACK_LOOKBACK_S))
    argv = ["journalctl", "--identifier", IDENTIFIER, "--since", stamp, "--output", "cat"]
    deadline = time.monotonic() + _READBACK_TIMEOUT_S
    while True:
        try:
            # Fixed argv, never a shell. `journalctl` is resolved off PATH rather than pinned to
            # an absolute path because its location differs across distros; nothing here is
            # caller-controlled, so there is nothing for a PATH to redirect that a compromised
            # agent could not already run as itself. It resolves under the stripped environment the
            # wrapper runs a probe in, too: with no PATH at all, exec falls back to the system
            # default (`/bin:/usr/bin`), where journalctl lives.
            done = subprocess.run(argv, capture_output=True, text=True, timeout=15, check=False)
        except FileNotFoundError as error:
            raise Unprovable("journalctl is not available on this box") from error
        except (OSError, subprocess.SubprocessError) as error:
            raise Unprovable(f"could not read the journal back: {error}") from error
        if done.returncode == 0 and all(message in done.stdout for message in messages):
            return True
        if done.returncode != 0 and time.monotonic() >= deadline:
            raise Unprovable(f"journalctl exited {done.returncode}: {done.stderr.strip()[:200]}")
        if time.monotonic() >= deadline:
            return False
        time.sleep(_READBACK_INTERVAL_S)


def main(argv: list[str] | None = None) -> int:
    """Emit one column's grammar into the journal and prove it landed.

    Usage: ``basecradle-harness-log-grammar <column>``. Three answers, per the claims contract:
    **0** the lines were rendered and are in the journal; **75** we could not determine (no
    journald, no ``journalctl``, an unknown column — a manifest naming something this build cannot
    exercise is a question we never got to ask); **any other non-zero** the lines were written and
    are not there, which is a real finding about this box.
    """
    args = sys.argv[1:] if argv is None else argv
    agent = agent_slug()
    try:
        if len(args) != 1:
            raise Unprovable(f"usage: {SCRIPT} <{'|'.join(COLUMNS)}>")
        column = args[0]
        if column not in COLUMNS:
            raise Unprovable(
                f"unknown column {column!r}; this build exercises {', '.join(COLUMNS)}"
            )
        messages = [record(line) for line in lines(column, agent=agent)]
        since = time.time()
        emit(messages)
        found = _read_back(messages, since=since)
    except Unprovable as error:
        print(f"UNPROVABLE: {error}", file=sys.stderr)
        return EX_TEMPFAIL
    if not found:
        print(
            f"FAILED: wrote {len(messages)} {column} line(s) under {IDENTIFIER} and could not "
            "read them back out of the journal",
            file=sys.stderr,
        )
        return 1
    print(f"proven: {column} ({len(messages)} clause lines) emitted as {agent} under {IDENTIFIER}")
    return 0


if __name__ == "__main__":  # pragma: no cover - console-script entry
    raise SystemExit(main())
