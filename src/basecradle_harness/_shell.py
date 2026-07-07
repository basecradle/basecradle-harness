"""Run a shell command: full command-line access as the agent's OS user.

The unlocked-profile counterpart to two tools an agent otherwise lacks or is fenced
away from. Where `code_execution` runs Python **in the vendor's own sandbox** and never
on the harness's box (issue #172), and `web_fetch` is a read-only HTTP GET behind an
SSRF fence, `shell` runs a **model-authored command line directly on the box**, as the
OS user the harness process runs as. It is the unguarded, human-equivalent version of
both: it can execute code locally (``python3 -c "…"``, run a ``.py`` or other script,
``pip install``, any interpreter present) **and** make arbitrary outbound network calls
(``curl``/``wget`` to any URL, method, and headers, with any credential the agent can
read from its environment).

Security model — the OS user's Unix permissions are the whole boundary
----------------------------------------------------------------------
There is **no per-command confirmation, no allow/deny-list, no fencing**. Commands run
as the agent's OS user, and *that user's Unix permissions are the sandbox* — exactly
what a human with an SSH shell on that account could do, no more and no less. That is
BaseCradle's human–AI parity applied to a shell: the AI peer gets the same terminal a
human peer would. The consequences, all intended and accepted:

- The agent can read its own env and secrets, and **read and modify its own harness
  code and its own guards** — anything its OS user can. This tool adds no fence of its
  own; the OS-user boundary *is* the fence.
- **This tool's safety rests entirely on the OS user being unprivileged** — no ``sudo``,
  not in ``docker``/``wheel``, not root. That is a *provisioning/deploy* invariant the
  box and the NOC verify before this tool is enabled, **not** something this tool
  enforces. **Never wire this onto a privileged account.**
- It runs **model-authored commands locally on the box** — a deliberate opt-out of the
  safe-default property that the shipped Harness executes no model code on its boxes
  (issue #172). That property is a safe-*default* (the locked profile), not an absolute;
  the unlocked profile is exactly where an operator opts out of it, and this tool is
  that opt-out — not a violation of #172.

Two gates, both required
------------------------
This is the one tool in the kit that needs **both** safety gates to clear, so a single
oversight can never arm it:

- It declares ``requires = {SHELL}``, so the shipped **locked** policy refuses it
  (`_apply_safe_policy` filters it out and surfaces the refusal in the Turn-0 brief). It
  survives only under `Policy.unlocked()`.
- Its plugin is ``opt_in`` (`_defaults/tools/shell.py`): off by default on every
  provider and dropped from the packaged fallback, so it loads only when an operator
  deliberately drops it into a persona's ``tools/`` overlay.

Implementation notes
--------------------
- **POSIX** — runs the command through ``/bin/bash -lc`` (a login shell, so the profile
  is sourced, matching a human's terminal). On a host without a POSIX ``/bin/bash`` the
  tool returns a clean "couldn't start a shell" error rather than crashing.
- **Bounded** — output is read on a background thread into a capped buffer, so a runaway
  producer (``yes``, ``cat /dev/zero``) can never balloon the harness's memory: past the
  cap the reader stops, the OS pipe fills, and the command is killed. A timed-out or
  cap-exceeded command is killed by its **process group** (`start_new_session=True`), so
  its children die with it — not just ``bash``.
- **Stateless (v1)** — each call is a fresh shell, so cwd, environment, and shell
  functions do **not** persist across calls (cwd defaults to ``workdir`` or the OS user's
  home each time). Persistent cwd is a possible later enhancement, not v1.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading

from basecradle_harness._policy import SHELL
from basecradle_harness._tools import Tool

# The default per-call timeout (seconds), and the hard ceiling a caller cannot exceed.
# A command past its timeout is killed; the ceiling stops the model from disabling the
# guard by asking for an enormous timeout.
DEFAULT_TIMEOUT = 120
MAX_TIMEOUT = 600

# The largest command output (characters) handed back to the model; a longer result is
# truncated at this cap with an explicit marker — never a silent cut. The reader stops
# reading once it passes the cap, so this bounds *memory*, not just the returned string.
MAX_OUTPUT = 30_000

# How long (seconds) to wait after killing a command's process group while draining its
# output / reaping it. Bounds the rare case of a detached grandchild that escaped the
# group and still holds the stdout pipe open, so nothing can hang the agent's turn.
DRAIN_TIMEOUT = 10


class ShellTool(Tool):
    """Run a shell command as the agent's OS user; return its output and exit code.

    Dangerous by construction: it declares ``requires = {SHELL}`` (refused by the locked
    profile) and ships opt-in. See the module docstring for the OS-user security model
    and the unprivileged-account requirement that this tool cannot enforce for itself.

    Args:
        workdir: The default directory commands run in when a call gives none. ``None``
            (the default) resolves to the OS user's home directory at call time.
        default_timeout: The per-call timeout (seconds) used when a call omits one.
            Clamped into ``[1, max_timeout]`` so a misconfiguration cannot disable the
            guard or make every command read as instantly timed-out.
        max_timeout: The hard ceiling a caller-supplied timeout is clamped to.
        max_output: The character cap on returned output (and on what is read into
            memory) before truncation.
        drain_timeout: How long to wait when draining/reaping a killed command.
    """

    name = "shell"
    description = (
        "Run a shell command as your OS user and get back its combined stdout+stderr and "
        "exit code. This is a real, unrestricted shell — the same access a human with a "
        "terminal on this account has. Two first-class uses: (1) execute code locally — "
        "'python3 -c \"…\"', run a .py or other script, 'pip install', any interpreter on "
        "the box; (2) make arbitrary outbound network calls — 'curl'/'wget' to any URL, "
        "method, and headers, with any credential you can read from your environment. Full "
        "shell syntax works (pipes, redirects, '&&', globs). There is no TTY, so "
        "interactive programs (vim, top, an ssh password prompt) will not work — pass "
        "everything on the command line. Each call is a fresh shell: the working directory "
        "and environment do NOT carry over between calls. Output is captured and very long "
        "output is truncated."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The command line to run, e.g. 'ls -la' or 'python3 script.py'.",
            },
            "timeout": {
                "type": "integer",
                "description": (
                    f"Seconds to allow before the command is killed "
                    f"(default {DEFAULT_TIMEOUT}, max {MAX_TIMEOUT})."
                ),
            },
            "workdir": {
                "type": "string",
                "description": "Directory to run the command in (default: your home directory).",
            },
        },
        "required": ["command"],
    }
    requires = frozenset({SHELL})

    def __init__(
        self,
        *,
        workdir: str | None = None,
        default_timeout: int = DEFAULT_TIMEOUT,
        max_timeout: int = MAX_TIMEOUT,
        max_output: int = MAX_OUTPUT,
        drain_timeout: int = DRAIN_TIMEOUT,
    ) -> None:
        self._workdir = workdir
        # Clamp the operator-set knobs so a misconfiguration can never weaken the guard:
        # the default timeout must be a real, positive value no larger than the ceiling.
        self._max_timeout = max(1, max_timeout)
        self._default_timeout = max(1, min(default_timeout, self._max_timeout))
        self._max_output = max(1, max_output)
        self._drain_timeout = max(1, drain_timeout)

    def run(
        self,
        command: str | None = None,
        timeout: int | None = None,
        workdir: str | None = None,
    ) -> str:
        """Run `command` in a fresh login shell; return its output and exit code.

        A non-zero exit is *reported* (in the ``[exit code: N]`` footer), never raised —
        a failing command is a normal result the model reads and reacts to, not a tool
        error. A command that outruns its timeout, or floods more than the output cap, is
        killed by its process group (children included) and reported as such.
        """
        if not command or not command.strip():
            return "Error: 'shell' needs a 'command' to run."

        cwd = workdir or self._workdir or os.path.expanduser("~")
        if not os.path.isdir(cwd):
            return f"Error: workdir {cwd!r} is not a directory."
        seconds = self._resolve_timeout(timeout)

        try:
            proc = subprocess.Popen(
                ["/bin/bash", "-lc", command],
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # merge stderr into stdout, preserving interleaving
                text=True,
                errors="replace",  # binary output must not crash the tool
                start_new_session=True,  # own process group so a timeout kills children too
            )
        except OSError as exc:
            return f"Couldn't start a shell: {exc}"

        # Read output on a background thread so we can bound BOTH how long we wait (the
        # timeout) and how much we buffer (the output cap) — `communicate()` can do
        # neither, and would OOM the harness on a flooding producer.
        pump = _OutputPump(proc.stdout, self._max_output)
        reader = threading.Thread(target=pump.run, daemon=True)
        reader.start()
        reader.join(seconds)

        if reader.is_alive():
            # Ran past the timeout without finishing or hitting the output cap → kill it.
            _kill_group(proc)
            reader.join(self._drain_timeout)  # bounded: an escaped grandchild can't hang us
            self._reap(proc)
            return self._format(pump.text(), timed_out=seconds)

        if pump.capped and proc.poll() is None:
            # Still producing after passing the output cap → over budget; stop it.
            _kill_group(proc)
            self._reap(proc)
            return self._format(pump.text(), over_cap=True)

        # Finished on its own (or produced more than the cap but already exited).
        self._reap(proc)
        return self._format(pump.text(), returncode=proc.returncode)

    def _resolve_timeout(self, timeout: int | None) -> int:
        """The effective timeout: the (clamped) default when unset/invalid, else clamped."""
        if timeout is None:
            return self._default_timeout
        try:
            seconds = int(timeout)
        except (TypeError, ValueError):
            return self._default_timeout
        return max(1, min(seconds, self._max_timeout))

    def _reap(self, proc: subprocess.Popen) -> None:
        """Wait for a finished-or-just-killed process, bounding the wait so it can't hang.

        The process has either exited on its own or just been process-group-killed, so
        this returns promptly; the bound only covers a pathological child that closed
        stdout yet lingers. A second kill + short wait is the last resort.
        """
        try:
            proc.wait(timeout=self._drain_timeout)
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            try:
                proc.wait(timeout=self._drain_timeout)
            except subprocess.TimeoutExpired:
                pass  # give up rather than block the turn; the child is a daemon's problem now

    def _format(
        self,
        output: str,
        *,
        returncode: int | None = None,
        timed_out: int | None = None,
        over_cap: bool = False,
    ) -> str:
        """Assemble the model-readable result: capped output + a one-line status footer."""
        body, truncated = self._cap(output or "")
        body = body.rstrip()  # drop the command's trailing newline so the footer sits flush
        lines = [body] if body else ["(no output)"]
        if over_cap:
            lines.append(f"[killed: output exceeded the {self._max_output}-character limit]")
        else:
            if truncated:
                lines.append(f"[output truncated at {self._max_output} characters]")
            if timed_out is not None:
                lines.append(f"[timed out after {timed_out}s and was killed]")
            else:
                lines.append(f"[exit code: {returncode}]")
        return "\n".join(lines)

    def _cap(self, output: str) -> tuple[str, bool]:
        """Truncate output to the cap; report whether anything was cut."""
        if len(output) <= self._max_output:
            return output, False
        return output[: self._max_output], True


class _OutputPump:
    """Read a process's stdout into a bounded buffer on a background thread.

    Reads in chunks and stops once it has passed `cap` characters — recording `capped` —
    so a runaway producer (``yes``, ``cat /dev/zero``) can never balloon the harness's
    memory: past the cap it stops reading, the OS pipe buffer fills, and the child blocks
    on write until it is killed. A command that finishes at or under the cap is read to
    EOF as normal. Runs on a daemon thread; a `read` interrupted by the stream closing
    (a killed process) just ends the pump.
    """

    _CHUNK = 8192

    def __init__(self, stream, cap: int) -> None:
        self._stream = stream
        self._cap = cap
        self._chunks: list[str] = []
        self.capped = False

    def run(self) -> None:
        total = 0
        try:
            while True:
                chunk = self._stream.read(self._CHUNK)
                if not chunk:
                    return  # EOF: the process closed stdout
                self._chunks.append(chunk)
                total += len(chunk)
                if total > self._cap:
                    self.capped = True
                    return  # stop reading; leave the rest in the pipe (throttles the child)
        except (ValueError, OSError):
            return  # the stream was closed under us (killed process) — stop

    def text(self) -> str:
        return "".join(self._chunks)


def _kill_group(proc: subprocess.Popen) -> None:
    """SIGKILL the command's whole process group, falling back to the process itself.

    ``start_new_session=True`` makes the child a new session/group **leader**, so the
    group's id equals ``proc.pid``. Signalling ``proc.pid`` as a group therefore hits the
    command and every child it spawned in one shot — and can never hit the harness's own
    group (whose id is a different pid), even in the brief window before ``setsid`` runs
    (there, no group with that id exists yet → `ProcessLookupError` → the process-only
    fallback). A bare ``proc.kill()`` would orphan the command's children.
    """
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        proc.kill()
