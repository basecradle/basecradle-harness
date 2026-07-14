#!/usr/bin/env python3
"""The version, the CHANGELOG, and the git tags must tell one story.

Both halves of this check exist because this repository shipped the failure each one now
forecloses. Neither failure broke anything, neither failed a test, and neither was visible
in a diff — which is the whole reason they are mechanical now instead of remembered.

**A published version erased from the CHANGELOG** (`repo` mode). A PR that bumped the version
renamed the *previous* release's header to its own new number and prepended its narrative to
that section. It happened twice — `v0.43.0` (#221) and `v0.47.0` (#240) — and both times a
version that was tagged and published on PyPI was left with no section at all, its content
credited to a version that was never released. A user running 0.43.0 who opened the changelog
to see what they had found nothing. The diff of such a PR looks like an ordinary new section;
only the whole file, read against the tag list, shows the erasure. `check_repo` fails that PR.

**A release cut while its headline fix still sat under `[Unreleased]`** (`preflight` mode,
issue #308). `[Unreleased]` is a legitimate staging area here — a PR may land content there
without bumping, and a later release commit promotes it — so this state is perfectly legal
mid-cycle and a lie the instant a tag is pushed. `check_release` runs before anything is built
and fails the release rather than publishing a wheel whose own changelog calls its headline fix
unreleased. It also pins the tag to `_version.py`: hatchling reads the version from the code,
not from the tag, so a `v0.72.0` tag on a commit that still says `0.71.0` publishes a wheel
under the wrong number entirely.

Usage:
    python scripts/check_release.py repo             # PR-time: the file and the tags agree
    python scripts/check_release.py preflight v1.2.3 # tag-time: this release is coherent
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHANGELOG = ROOT / "CHANGELOG.md"
VERSION_FILE = ROOT / "src" / "basecradle_harness" / "_version.py"

UNRELEASED = "Unreleased"
_HEADER = re.compile(r"^## \[([^\]]+)\]", re.M)
_VERSION = re.compile(r'__version__\s*=\s*"([^"]+)"')
_NUMERIC = re.compile(r"^(\d+(?:\.\d+)*)(.*)$")


def version(text: str) -> str:
    """The version the build backend will publish — read from the code, as hatchling does."""
    match = _VERSION.search(text)
    if not match:
        raise ValueError(f"no __version__ found in {VERSION_FILE}")
    return match.group(1)


def headers(changelog: str) -> list[str]:
    """Every `## [x]` header, in file order (newest first), `Unreleased` included."""
    return _HEADER.findall(changelog)


def released(changelog: str) -> list[str]:
    """Only the version headers — the ones that claim a release happened."""
    return [h for h in headers(changelog) if h != UNRELEASED]


def unreleased_body(changelog: str) -> str:
    """Whatever sits under `## [Unreleased]`, up to the next section."""
    parts = re.split(r"^## \[Unreleased\][^\n]*\n", changelog, maxsplit=1, flags=re.M)
    if len(parts) < 2:
        return ""
    return re.split(r"^## \[", parts[1], maxsplit=1, flags=re.M)[0].strip()


def sort_key(v: str) -> tuple[tuple[int, ...], int, str]:
    """A total order over version strings — it never raises, whatever a tag turns out to look like.

    A guard that crashes on an input it did not anticipate is worse than the bug it was written
    to catch. `v1.0.0rc1` is a legal PEP 440 tag and `release.yml` fires on every `v*`, so an
    `int("0rc1")` here would not merely fail one release — the tag does not go away, so every
    later PR's `changelog` job would crash on it too, and CI would be bricked repo-wide by the
    check that exists to protect it.

    Numeric parts order numerically; a pre-release suffix sorts *before* its final release
    (`1.0.0rc1` < `1.0.0`), which is what puts the sections in the right order in a newest-first
    changelog.
    """
    match = _NUMERIC.match(v)
    if not match:
        return ((), 0, v)
    numbers = tuple(int(part) for part in match.group(1).split("."))
    suffix = match.group(2)
    return (numbers, 0 if suffix else 1, suffix)


def git_tags() -> list[str]:
    """The `v*` tags, as the repository actually has them."""
    out = subprocess.run(
        ["git", "tag", "--list", "v*"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return sorted(out.stdout.split(), key=lambda t: sort_key(t.removeprefix("v")))


def check_repo(changelog: str, code: str, tags: list[str]) -> list[str]:
    """Invariants true of `main` at every commit — not only at a release."""
    errors: list[str] = []
    sections = released(changelog)
    coded = version(code)

    # A check that cannot find its input is a no-op guard that reads green forever. In CI this
    # means `fetch-depth: 0`; the shallow default fetches no tags, and this check would then
    # "pass" by having nothing to check.
    if not tags:
        errors.append(
            "no v* git tags found — this check needs the full history "
            "(actions/checkout with `fetch-depth: 0`). Refusing to pass vacuously."
        )

    for tag in tags:
        tagged = tag.removeprefix("v")
        if tagged not in sections:
            errors.append(
                f"{tag} is tagged but has no `## [{tagged}]` section in CHANGELOG.md. "
                "A released version must keep its own section — never fold it into another's."
            )

    for dupe in sorted({s for s in sections if sections.count(s) > 1}):
        errors.append(f"CHANGELOG.md has more than one `## [{dupe}]` section.")

    if sections != sorted(sections, key=sort_key, reverse=True):
        errors.append("CHANGELOG.md version sections are not in descending order.")

    # Every bump ships with its section, and every section ships with its bump. (Content staged
    # under `[Unreleased]` is exempt by construction — it names no version yet.)
    if sections and sections[0] != coded:
        errors.append(
            f"_version.py says {coded} but the newest CHANGELOG section is "
            f"[{sections[0]}]. A version bump and its section land together."
        )

    return errors


def check_release(changelog: str, code: str, tag: str, tags: list[str]) -> list[str]:
    """Everything above, plus what only a tag can be wrong about.

    The tag list is passed in rather than read from git here, so that this stays a pure function
    of what it is handed — the caller decides where the tags come from, and a test never depends
    on the state of the repository it happens to be running inside.
    """
    errors = check_repo(changelog, code, tags)
    tagged, coded = tag.removeprefix("v"), version(code)

    if tagged != coded:
        errors.append(
            f"tag {tag} does not match _version.py ({coded}). The build reads the version from "
            f"the code, so this tag would publish a {coded} wheel."
        )
    if tagged not in released(changelog):
        errors.append(
            f"no `## [{tagged}]` section in CHANGELOG.md — releasing an undocumented version."
        )

    staged = unreleased_body(changelog)
    if staged:
        errors.append(
            "[Unreleased] is not empty — promote it into this release's section or leave it out "
            f"of the release deliberately. Cutting a tag over staged changes publishes a version "
            f"whose own changelog calls them unreleased (issue #308). It currently holds:\n"
            f"    {staged.splitlines()[0][:88]}"
        )

    return errors


def main(argv: list[str]) -> int:
    try:
        changelog, code = CHANGELOG.read_text(), VERSION_FILE.read_text()
        tags = git_tags()

        if argv == ["repo"]:
            errors = check_repo(changelog, code, tags)
            ok = f"CHANGELOG.md, _version.py ({version(code)}), and {len(tags)} tags agree."
        elif len(argv) == 2 and argv[0] == "preflight":
            errors = check_release(changelog, code, argv[1], tags)
            ok = f"{argv[1]} is coherent: version, CHANGELOG, and tag tell one story."
        else:
            print(__doc__)
            return 2
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        # The checkers are pure functions of what they are handed; only main touches the world,
        # so only main has anything to fail on.
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    if errors:
        return 1
    print(ok)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
