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

import importlib.util
from pathlib import Path

import pytest

# The one reader of a workflow's `uses:` lines lives beside the job that checks each pin's comment
# against its action's repository (issue #916), so the shape this guard checks and the truth that
# job checks are read off the same set of pins.
_SPEC = importlib.util.spec_from_file_location(
    "check_action_pins", Path(__file__).resolve().parent.parent / "scripts" / "check_action_pins.py"
)
assert _SPEC is not None and _SPEC.loader is not None
pins = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pins)

PINNED = pins.PINNED
VERSION_COMMENT = pins.VERSION_COMMENT
references = pins.references
third_party = pins.third_party


def test_the_guard_sees_the_publish_jobs() -> None:
    """A scan that matched nothing would pass every rule below, so it must find what it guards."""
    refs = {ref for name, _, ref, _ in third_party() if name == "workflows/release.yml"}
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


def test_one_sha_names_one_release() -> None:
    """A hand edit that moved a SHA in one place and not its comment leaves two names for it."""
    releases: dict[str, set[str]] = {}
    for _, _, ref, rest in third_party():
        releases.setdefault(ref, set()).add(rest.strip())
    assert {ref: names for ref, names in releases.items() if len(names) > 1} == {}


def test_the_exemptions_cover_only_what_they_name() -> None:
    """An exempt reference is either local or the fleet's own reusable workflow, nothing wider."""
    for name, number, ref, _ in references():
        if ref.startswith("basecradle/"):
            assert ref.startswith("basecradle/.github/.github/workflows/"), (
                f"{name}:{number}: `{ref}` is exempt only as a fleet reusable workflow"
            )


# The live half (`scripts/check_action_pins.py`, CI's `pins` job) reads each tag off its action's
# repository. Its reading of `git ls-remote` is checked here, offline, on fabricated output.
_LIGHTWEIGHT = "3d3c42e5aac5ba805825da76410c181273ba90b1\trefs/tags/v7.0.1\n"
_ANNOTATED = (
    "aef6c9b6e301dc803e1bc9133e9ce67ae9b0906b\trefs/tags/v1.14.3\n"
    "7245906fb61fa33a13554f9d5fb856fe61758086\trefs/tags/v1.14.3^{}\n"
)


def test_a_lightweight_tag_names_its_commit() -> None:
    assert pins.tagged_commit(_LIGHTWEIGHT, "v7.0.1") == "3d3c42e5aac5ba805825da76410c181273ba90b1"


def test_an_annotated_tag_is_followed_to_its_commit() -> None:
    """A SHA pin names the commit, never the tag object an annotated tag points at first."""
    assert pins.tagged_commit(_ANNOTATED, "v1.14.3") == "7245906fb61fa33a13554f9d5fb856fe61758086"


def test_a_missing_tag_names_nothing() -> None:
    assert pins.tagged_commit("", "v1.14.3") is None
    assert pins.tagged_commit(_ANNOTATED, "v1.14") is None
