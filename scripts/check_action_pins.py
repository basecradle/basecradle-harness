"""Every pinned action's version comment names the release its SHA is (issue #916).

An action is pinned as ``owner/action@<40-hex sha> # vX.Y.Z`` (issue #914). The SHA is what runs;
the comment is what a reviewer reads, and what Dependabot rewrites with the SHA. A hand edit that
moved one and not the other leaves a pin whose comment says a release it is not, and nothing
offline can tell: whether ``v7.0.1`` is ``3d3c42e…`` is a fact only the action's repository holds.
So this is its own CI job (``pins``), and the offline suite never runs it.

This module is also the one reader of a workflow's ``uses:`` lines: ``tests/test_workflow_pins.py``
imports :func:`references` and :func:`third_party` from here, so the guard that checks a pin's
shape and the job that checks its truth can never read two different sets of pins.

Standard library only, run with the runner's own ``python3``. Each tag is read with
``git ls-remote`` over HTTPS, which needs no token for a public repository and spends no REST API
rate limit. An annotated tag is followed to its commit (the ``^{}`` line), which is what a SHA pin
names.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GITHUB = ROOT / ".github"

# Any spelling of the key YAML accepts, quoted or with a space before the colon, so a line the
# parser below cannot read fails the build instead of being skipped.
MENTIONS_USES = re.compile(r"""\buses["']?\s*:""")

USES = re.compile(r"^\s*(?:-\s+)?uses:\s*(?P<ref>\S+)(?P<rest>.*)$")
PINNED = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:/\S*)?@[0-9a-f]{40}$")
VERSION_COMMENT = re.compile(r"^\s+#\s+(?P<release>v\d+\.\d+\.\d+)\s*$")
EXEMPT = ("./", "basecradle/")

ATTEMPTS = 3


class UnreadableUses(ValueError):
    """A line names the ``uses`` key in a form this reader cannot parse."""


def workflows() -> list[Path]:
    """Every file the runner reads a ``uses:`` from: the workflows, and any composite action."""
    return sorted(
        [*(GITHUB / "workflows").glob("*.y*ml"), *(GITHUB / "actions").glob("**/action.y*ml")]
    )


def references() -> list[tuple[str, int, str, str]]:
    """Every ``uses:`` in every workflow, as (file, line number, reference, rest of line)."""
    found = []
    for path in workflows():
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if not MENTIONS_USES.search(line) or line.lstrip().startswith("#"):
                continue
            match = USES.match(line)
            if not match:
                raise UnreadableUses(
                    f"{path.name}:{number}: a `uses:` line this guard cannot read: {line!r}"
                )
            found.append((str(path.relative_to(GITHUB)), number, match["ref"], match["rest"]))
    return found


def third_party() -> list[tuple[str, int, str, str]]:
    return [ref for ref in references() if not ref[2].startswith(EXEMPT)]


def tagged_commit(listing: str, tag: str) -> str | None:
    """The commit ``tag`` names in ``git ls-remote`` output, an annotated tag followed to it."""
    refs = {}
    for line in listing.splitlines():
        sha, _, ref = line.partition("\t")
        refs[ref.strip()] = sha.strip()
    return refs.get(f"refs/tags/{tag}^{{}}") or refs.get(f"refs/tags/{tag}")


def ls_remote(repository: str, tag: str) -> str:
    """``git ls-remote`` for one tag and its peeled commit, retried on a transport failure."""
    command = [
        "git",
        "ls-remote",
        "--tags",
        f"https://github.com/{repository}.git",
        f"refs/tags/{tag}",
        f"refs/tags/{tag}^{{}}",
    ]
    # No prompt for a repository that does not exist (GitHub answers it as an auth challenge).
    environment = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    for attempt in range(1, ATTEMPTS + 1):
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=60, env=environment, check=False
            )
        except subprocess.TimeoutExpired:
            # A hung transport is the case a retry is for.
            failure = "timed out after 60s"
        else:
            if result.returncode == 0:
                return result.stdout
            failure = result.stderr.strip()
        if attempt == ATTEMPTS:
            raise RuntimeError(f"git ls-remote {repository} {tag}: {failure}")
        time.sleep(5 * attempt)
    raise AssertionError("unreachable")


def main() -> int:
    pins: dict[tuple[str, str, str], list[str]] = {}
    for name, number, ref, rest in third_party():
        action, _, sha = ref.partition("@")
        comment = VERSION_COMMENT.match(rest)
        if not PINNED.match(ref) or not comment:
            # Shape is the offline guard's to report (tests/test_workflow_pins.py).
            print(f"::error file=.github/{name},line={number}::`{ref}` is not a commented SHA pin")
            return 1
        repository = "/".join(action.split("/")[:2])
        pins.setdefault((repository, sha, comment["release"]), []).append(f"{name}:{number}")

    failures = 0
    for (repository, sha, release), sites in sorted(pins.items()):
        commit = tagged_commit(ls_remote(repository, release), release)
        if commit == sha:
            print(f"ok {repository}@{sha} is {release}")
            continue
        failures += 1
        actual = f"is {commit}" if commit else "does not exist"
        for site in sites:
            name, _, number = site.rpartition(":")
            print(
                f"::error file=.github/{name},line={number}::`{repository}@{sha}` says "
                f"{release}, but {repository}'s {release} {actual}"
            )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
