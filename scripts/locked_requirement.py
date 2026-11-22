"""Print one package of ``uv.lock`` as a hash-pinned requirement (issue #916).

The release build installs uv itself with this, before any uv exists to read the lock:
``pip install --require-hashes --no-deps --only-binary :all: -r <file>`` then accepts only a wheel
whose sha256 the lock records, so the uv that builds the published bytes is held by a hash in
this repository's own reviewed diff, and Dependabot's uv entry moves it. ``astral-sh/setup-uv``
cannot give that: it downloads uv's GitHub release archive, and for a uv newer than the action's
built-in checksum table it checks that archive against a checksum fetched at run time from the
same host.

Standard library only, run with the runner's own ``python3``::

    python3 scripts/locked_requirement.py uv
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

LOCK = Path(__file__).resolve().parent.parent / "uv.lock"


def _hashes(package: dict) -> list[str]:
    files = [*package.get("wheels", []), *([package["sdist"]] if "sdist" in package else [])]
    return [file["hash"] for file in files]


def locked_package(lock: dict, name: str) -> dict:
    """The lock's one PyPI entry for ``name``, every file of it sha256-pinned, or a ValueError.

    One rule for every reader that must know which entry is meant: the uv bootstrap below, and
    the walk ``tests/test_dependabot.py`` makes of the build toolchain.
    """
    packages = [package for package in lock.get("package", []) if package.get("name") == name]
    if len(packages) != 1:
        raise ValueError(f"uv.lock holds {len(packages)} `{name}` packages, not one")
    (package,) = packages
    if package.get("source") != {"registry": "https://pypi.org/simple"}:
        raise ValueError(f"uv.lock's `{name}` is not from PyPI: {package.get('source')}")
    hashes = _hashes(package)
    if not hashes or not all(value.startswith("sha256:") for value in hashes):
        raise ValueError(f"uv.lock's `{name}` carries no sha256 for every file")
    return package


def requirement(lock: dict, name: str) -> str:
    """``name==<locked version>`` with every hash the lock holds for it, or a ValueError."""
    package = locked_package(lock, name)
    hashes = _hashes(package)
    return " ".join([f"{name}=={package['version']}", *(f"--hash={value}" for value in hashes)])


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: locked_requirement.py <package>", file=sys.stderr)
        return 2
    try:
        print(requirement(tomllib.loads(LOCK.read_text()), argv[0]))
    except ValueError as error:
        print(f"locked_requirement: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
