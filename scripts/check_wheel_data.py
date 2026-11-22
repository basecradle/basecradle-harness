#!/usr/bin/env python3
"""The built wheel carries every data file the package publishes, byte for byte (issue #997).

A data file the package documents as public is a promise to readers who never import the harness:
they open the released wheel, or the installed package, at a stated path. Hatchling ships every
file under the package directory, so a file goes missing from a wheel only by something nobody
sees in a diff — an ignore rule, a build-target change, a rename — and the tests, which run from
the source tree, would still pass. So CI's build job runs this against the wheel the release's own
build action produced, and the release builds the same tree with the same action.

Usage:
    python3 scripts/check_wheel_data.py dist    # the directory the build wrote
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Every data file the package publishes, by its path inside the wheel. Its source is the same path
#: under ``src/``.
PUBLISHED = ("basecradle_harness/data/prices.json",)


def check(wheel: Path, root: Path = ROOT) -> list[str]:
    """Every way `wheel` fails to carry the published data files as the source tree holds them."""
    errors = []
    with zipfile.ZipFile(wheel) as archive:
        members = set(archive.namelist())
        for path in PUBLISHED:
            if path not in members:
                errors.append(f"{wheel.name} carries no {path}")
                continue
            if archive.read(path) != (root / "src" / path).read_bytes():
                errors.append(f"{wheel.name}'s {path} differs from src/{path}")
    return errors


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    wheels = sorted(Path(argv[0]).glob("*.whl"))
    if len(wheels) != 1:
        print(f"expected one wheel in {argv[0]}, found {len(wheels)}", file=sys.stderr)
        return 1
    errors = check(wheels[0])
    for error in errors:
        print(error, file=sys.stderr)
    if not errors:
        print(f"{wheels[0].name} carries {', '.join(PUBLISHED)}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
