"""Every third-party action this repository runs is pinned to a commit SHA (issue #914).

A tag (``@v7``) or a branch (``@release/v1``) is a pointer its owner can move, so a workflow that
names one runs whatever the pointer says on the day it runs. The publish jobs hold
``id-token: write``, and a moved ref there would publish this package under our name. The rule
is applied to every workflow, not only to the jobs that can publish or that hold a secret: a new
job that gains a secret or a write permission is then covered without its author having to
classify it first, and Dependabot keeps every pin current at no extra cost.

The pin is a 40-character commit SHA with the release it stands for in a comment on the same
line, the shape Dependabot's github-actions updater rewrites (it moves the SHA and the comment
together, and a SHA that belongs to a release tag is moved to the next release, not to a branch
head).

Two kinds of reference are exempt, each for a stated reason:

* ``./`` — an action in this repository, which moves only with our own commits.
* ``basecradle/`` — the fleet's own reusable workflows, called by the shared ``needs-human``
  alert and Dependabot auto-merge stubs. They are first-party, and the stubs are carried byte
  for byte from the capital, so pinning them is a fleet decision made there, never an edit here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = sorted((ROOT / ".github" / "workflows").glob("*.y*ml"))

USES = re.compile(r"^\s*(?:-\s+)?uses:\s*(?P<ref>\S+)(?P<rest>.*)$")
PINNED = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:/\S*)?@[0-9a-f]{40}$")
VERSION_COMMENT = re.compile(r"^\s+#\s+v\d+\.\d+\.\d+\s*$")
EXEMPT = ("./", "basecradle/")


def references() -> list[tuple[str, int, str, str]]:
    """Every ``uses:`` in every workflow, as (file, line number, reference, rest of line)."""
    found = []
    for path in WORKFLOWS:
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if "uses:" not in line or line.lstrip().startswith("#"):
                continue
            match = USES.match(line)
            assert match, f"{path.name}:{number}: a `uses:` line this guard cannot read: {line!r}"
            found.append((path.name, number, match["ref"], match["rest"]))
    return found


def third_party() -> list[tuple[str, int, str, str]]:
    return [ref for ref in references() if not ref[2].startswith(EXEMPT)]


def test_the_guard_sees_the_publish_jobs() -> None:
    """A scan that matched nothing would pass every rule below, so it must find what it guards."""
    refs = {ref for name, _, ref, _ in third_party() if name == "release.yml"}
    assert any(ref.startswith("pypa/gh-action-pypi-publish@") for ref in refs)
    assert len(third_party()) >= 10


@pytest.mark.parametrize(("name", "number", "ref", "rest"), third_party())
def test_every_third_party_action_is_pinned_to_a_commit_sha(
    name: str, number: int, ref: str, rest: str
) -> None:
    assert PINNED.match(ref), (
        f"{name}:{number}: `{ref}` is a moving reference. Pin it to the full commit SHA of a "
        "release, read off the action's own repository, with the release in a comment: "
        "`owner/action@<40-hex sha> # vX.Y.Z`."
    )


@pytest.mark.parametrize(("name", "number", "ref", "rest"), third_party())
def test_every_pin_names_its_release(name: str, number: int, ref: str, rest: str) -> None:
    """Dependabot rewrites a same-line version comment; without one a pin is an opaque hash."""
    assert VERSION_COMMENT.match(rest), (
        f"{name}:{number}: `{ref}` carries no `# vX.Y.Z` comment naming the release it pins."
    )


def test_the_exemptions_cover_only_what_they_name() -> None:
    """An exempt reference is either local or the fleet's own reusable workflow, nothing wider."""
    for name, number, ref, _ in references():
        if ref.startswith("basecradle/"):
            assert ref.startswith("basecradle/.github/.github/workflows/"), (
                f"{name}:{number}: `{ref}` is exempt only as a fleet reusable workflow"
            )
