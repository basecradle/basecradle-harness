"""Nothing that makes the published bytes resolves fresh on the day of a tag (issue #916).

Pinning every action to a SHA (issue #914, ``tests/test_workflow_pins.py``) fixes which actions
run, not what they fetch. Whatever builds the wheel is as powerful as whatever uploads it, so each
link of the build is held by ``uv.lock``, where Dependabot moves it as a reviewable diff and the
daily dependency scan reads it:

* **uv** is the ``build`` group's. The build installs the very wheel the lock holds, by pip in
  hash-checking mode from the lock's own sha256s (``scripts/locked_requirement.py``), and every
  ``astral-sh/setup-uv`` step in CI reads the same version off the lock (``version-file:
  uv.lock``). Given no version, setup-uv installs the newest uv; given a ``version:`` it installs
  what the workflow says and the lock no longer decides.
* **The build backend** and everything it imports are exported from the lock as hashed build
  constraints, and ``uv build --require-hashes`` refuses any build requirement without one. Every
  name in ``[build-system].requires`` is declared in the ``build`` group with the same specifier,
  or it would never reach the lock.
* **The interpreter** is uv-managed (``--managed-python``), so it is what the pinned uv knows and
  verifies, never the runner image's.

There is one build, ``.github/actions/build-distributions``, used by the release and by CI alike,
and CI's gate requires it, so a bump is built on its pull request before it can build a release.
The guards here read the workflows as text (this repository carries no YAML parser, and adds none
for a test); each one is anchored by a check that it found what it guards, so a reader that matched
nothing fails rather than passes.
"""

from __future__ import annotations

import importlib.util
import re
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parent.parent
GITHUB = ROOT / ".github"
BUILD_ACTION = GITHUB / "actions" / "build-distributions" / "action.yml"
RELEASE = GITHUB / "workflows" / "release.yml"
CI = GITHUB / "workflows" / "ci.yml"
ACTIONS = sorted((GITHUB / "actions").glob("**/action.y*ml"))
FILES = sorted([*(GITHUB / "workflows").glob("*.y*ml"), *ACTIONS])

PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())
LOCK = tomllib.loads((ROOT / "uv.lock").read_text())

_SPEC = importlib.util.spec_from_file_location(
    "locked_requirement", ROOT / "scripts" / "locked_requirement.py"
)
assert _SPEC is not None and _SPEC.loader is not None
locked_requirement = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(locked_requirement)

STEP_USES = re.compile(r"^(?P<indent>\s*)(?:-\s+)?uses:\s*(?P<ref>\S+)")
# A block scalar's header: `|` or `>`, with an optional chomping and indentation indicator.
BLOCK_SCALAR = re.compile(r"^[|>](?:[+-]?\d?|\d[+-]?)$")


def step(lines: list[str], at: int) -> list[str]:
    """The lines of the workflow step whose ``uses:`` is on line ``at`` (0-based).

    A step is the ``- `` item holding the line: back to the dash at the step's own indentation,
    forward to the next line at that indentation or shallower that is not blank or a comment.
    """
    line = lines[at]
    dash = line.index("-") if line.lstrip().startswith("-") else None
    if dash is None:
        key = len(line) - len(line.lstrip())
        dash = key - 2
        start = at
        while start > 0 and not (
            lines[start].lstrip().startswith("- ")
            and len(lines[start]) - len(lines[start].lstrip()) == dash
        ):
            start -= 1
    else:
        start = at
    end = start + 1
    while end < len(lines):
        text = lines[end]
        stripped = text.lstrip()
        if stripped and not stripped.startswith("#") and len(text) - len(stripped) <= dash:
            break
        end += 1
    return lines[start:end]


def uses_in(lines: list[str]) -> list[str]:
    """Every ``uses:`` reference on the given lines, commented-out lines excluded."""
    return [
        match["ref"]
        for line in lines
        if not line.lstrip().startswith("#") and (match := STEP_USES.match(line))
    ]


def steps_using(prefix: str) -> list[tuple[str, int, list[str]]]:
    """Every step in every workflow and composite action whose ``uses:`` starts with ``prefix``."""
    found = []
    for path in FILES:
        lines = path.read_text().splitlines()
        for index, line in enumerate(lines):
            match = STEP_USES.match(line)
            if match and not line.lstrip().startswith("#") and match["ref"].startswith(prefix):
                found.append((str(path.relative_to(GITHUB)), index + 1, step(lines, index)))
    return found


def setting(block: list[str], key: str) -> str | None:
    """The value of ``key:`` anywhere in a step's lines, comments ignored."""
    for line in block:
        match = re.match(rf"^\s*{re.escape(key)}:\s*(?P<value>[^#]*?)\s*(?:#.*)?$", line)
        if match:
            return match["value"].strip("\"'")
    return None


def job(workflow: Path, name: str) -> list[str]:
    """The lines of one job, by its key under ``jobs:``."""
    text = workflow.read_text()
    jobs = re.findall(rf"^ {{2}}{re.escape(name)}:\n(?:(?: {{4,}}.*)?\n)+", text, re.MULTILINE)
    assert len(jobs) == 1, f"{workflow.name} has one `{name}` job"
    return jobs[0].splitlines()


def commands(text: str) -> list[str]:
    """Each ``run:`` of a file as one line, a block scalar's lines joined with single spaces."""
    found = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        match = re.match(r"^(?P<indent>\s*)(?:-\s+)?run:\s*(?P<body>.*)$", line)
        if not match or line.lstrip().startswith("#"):
            continue
        body = match["body"]
        if BLOCK_SCALAR.match(body):
            indent = len(match["indent"])
            parts = []
            for following in lines[index + 1 :]:
                if following.strip() and len(following) - len(following.lstrip()) <= indent:
                    break
                parts.append(following.strip())
            body = " ".join(part for part in parts if part)
        found.append(body)
    return found


def build_group() -> dict[str, Requirement]:
    return {
        Requirement(item).name: Requirement(item)
        for item in PYPROJECT["dependency-groups"]["build"]
    }


def locked(name: str) -> list[dict]:
    return [package for package in LOCK["package"] if package["name"] == name]


def test_the_command_reader_reads_every_block_scalar() -> None:
    """A header it did not know would make the guards below read the header, never the command."""
    text = "\n".join(
        f"      - run: {header}\n          uv build\n          --flag"
        for header in ("|+", ">2", "|")
    )
    assert commands(text) == ["uv build --flag"] * 3


# --- uv --------------------------------------------------------------------------------------


def test_the_lock_holds_one_exact_uv() -> None:
    """setup-uv takes the first ``uv`` package in the lock, and the build installs that one."""
    assert "uv" in build_group()
    (package,) = locked("uv")
    assert re.fullmatch(r"\d+\.\d+\.\d+", package["version"])
    requirement = locked_requirement.requirement(LOCK, "uv")
    assert requirement.startswith(f"uv=={package['version']} --hash=sha256:")
    assert requirement.count("--hash=") == len(package["wheels"]) + ("sdist" in package)


def test_the_build_installs_uv_by_its_locked_hashes() -> None:
    (install,) = [command for command in commands(BUILD_ACTION.read_text()) if "pip " in command]
    assert "python3 scripts/locked_requirement.py uv >" in install
    for flag in ("--require-hashes", "--no-deps", "--only-binary :all:"):
        assert flag in install, f"the uv bootstrap lost `{flag}`"


@pytest.mark.parametrize(
    ("lock", "fault"),
    [
        ({"package": []}, "holds 0"),
        ({"package": [{"name": "uv"}, {"name": "uv"}]}, "holds 2"),
        ({"package": [{"name": "uv", "source": {"git": "https://example.com/uv"}}]}, "not from"),
        (
            {
                "package": [
                    {
                        "name": "uv",
                        "version": "1.0.0",
                        "source": {"registry": "https://pypi.org/simple"},
                        "wheels": [{"hash": "md5:0f343b0931126a20f133d67c2b018a3b"}],
                    }
                ]
            },
            "no sha256",
        ),
    ],
)
def test_a_lock_that_cannot_pin_uv_refuses(lock: dict, fault: str) -> None:
    with pytest.raises(ValueError, match=fault):
        locked_requirement.requirement(lock, "uv")


def test_the_guard_sees_every_setup_uv() -> None:
    """CI's lint, test and mempalace jobs: a reader that found none would pass every rule."""
    sites = {name for name, _, _ in steps_using("astral-sh/setup-uv@")}
    assert sites == {"workflows/ci.yml"}
    assert len(steps_using("astral-sh/setup-uv@")) >= 3


@pytest.mark.parametrize(("name", "number", "block"), steps_using("astral-sh/setup-uv@"))
def test_every_setup_uv_installs_the_locked_uv(name: str, number: int, block: list[str]) -> None:
    assert setting(block, "version-file") == "uv.lock", (
        f"{name}:{number}: setup-uv without `version-file: uv.lock` installs the newest uv, or the "
        "floor pyproject.toml declares, on the day it runs"
    )
    assert setting(block, "version") is None, (
        f"{name}:{number}: a `version:` input outranks the lock, so CI and the release can disagree"
    )


# --- the build backend -----------------------------------------------------------------------


def test_every_build_requirement_is_locked_with_its_own_specifier() -> None:
    """A name in ``[build-system].requires`` that the group lacks resolves fresh on every tag."""
    requires = [Requirement(item) for item in PYPROJECT["build-system"]["requires"]]
    assert requires, "the guard must see the build backend"
    group = build_group()
    for requirement in requires:
        assert requirement.name in group, (
            f"`{requirement}` is a build requirement the `build` group does not lock"
        )
        assert group[requirement.name].specifier == requirement.specifier, (
            f"the `build` group's `{group[requirement.name]}` and `[build-system]`'s "
            f"`{requirement}` are two statements of one requirement"
        )
        (package,) = locked(requirement.name)
        assert requirement.specifier.contains(package["version"]), (
            f"uv.lock holds {requirement.name} {package['version']}, outside `{requirement}`"
        )
        assert package.get("wheels") or package.get("sdist"), "a locked package carries hashes"


def test_the_build_action_builds_from_the_locked_hashes() -> None:
    found = commands(BUILD_ACTION.read_text())
    assert any(
        "uv export --quiet --locked --only-group build --no-emit-project" in c for c in found
    )
    (build,) = [command for command in found if command.startswith("uv build")]
    for flag in ("--build-constraints", "--require-hashes", "--managed-python", "--no-cache"):
        assert flag in build, f"the release build lost `{flag}`"


def test_nothing_builds_but_the_build_action() -> None:
    """A second build (``uv build``, ``python -m build``…) is one the constraints do not reach."""
    builder = re.compile(
        r"\buv build\b|-m build\b|\bpyproject-build\b|\bpip wheel\b|\bhatch build\b"
    )
    for path in FILES:
        for command in commands(path.read_text()):
            if builder.search(command):
                assert path == BUILD_ACTION, f"{path.name} builds outside the build action"


def test_the_release_build_job_runs_only_the_one_build() -> None:
    """Its checkout, the build, and the upload of what it built: no third-party build action."""
    refs = [ref.partition("@")[0] for ref in uses_in(job(RELEASE, "build"))]
    assert refs == [
        "actions/checkout",
        "./.github/actions/build-distributions",
        "actions/upload-artifact",
    ]


def test_ci_builds_with_the_one_build_and_requires_it() -> None:
    """A CI build its gate does not need would run, fail, and block nothing."""
    assert [ref.partition("@")[0] for ref in uses_in(job(CI, "build"))] == [
        "actions/checkout",
        "./.github/actions/build-distributions",
    ]
    (needs,) = [line for line in job(CI, "ci") if line.strip().startswith("needs:")]
    gate = {name.strip() for name in needs.split("[", 1)[1].rstrip("]").split(",")}
    assert {"build", "pins"} <= gate


def test_the_release_job_installs_nothing_of_its_own() -> None:
    """Every tool the release runs is an action pinned to a SHA or a byte the lock holds.

    The one install allowed is the uv bootstrap, pip in hash-checking mode with no dependencies
    and no source builds, from the hashes ``uv.lock`` records.
    """
    installer = re.compile(
        r"\b(?:pip3? install|uv (?:tool |pip )?install|uvx|curl|wget|apt(?:-get)?|npm|npx)\b"
    )
    for path in (RELEASE, BUILD_ACTION):
        for command in commands(path.read_text()):
            if not installer.search(command):
                continue
            assert re.search(r"-m pip install\b", command), f"{path.name} installs: `{command}`"
            for flag in ("--require-hashes", "--no-deps", "--only-binary :all:"):
                assert flag in command, f"{path.name} installs without `{flag}`: `{command}`"


# --- what Dependabot can move ----------------------------------------------------------------


def test_the_guard_sees_the_build_action() -> None:
    assert BUILD_ACTION in ACTIONS


@pytest.mark.parametrize("path", ACTIONS, ids=lambda path: str(path.relative_to(GITHUB)))
def test_no_composite_action_names_a_third_party_action(path: Path) -> None:
    """Dependabot's github-actions entry reads the workflows, never ``.github/actions/``, so a pin
    there would be left behind by every bump of the same action in the workflows."""
    third_party = [
        ref for ref in uses_in(path.read_text().splitlines()) if not ref.startswith("./")
    ]
    assert third_party == [], f"{path.name} pins what Dependabot never moves: {third_party}"


def test_one_action_is_pinned_to_one_sha() -> None:
    """Two SHAs for one action is a bump that moved one site and left another behind."""
    shas: dict[str, set[str]] = {}
    for path in FILES:
        for ref in uses_in(path.read_text().splitlines()):
            action, _, sha = ref.partition("@")
            if not ref.startswith(("./", "basecradle/")):
                shas.setdefault(action, set()).add(sha)
    assert shas, "the guard must see the pins"
    assert {action: found for action, found in shas.items() if len(found) > 1} == {}


# --- checkouts -------------------------------------------------------------------------------


def test_the_guard_sees_every_checkout() -> None:
    assert len(steps_using("actions/checkout@")) >= 8
    assert any(name == "workflows/release.yml" for name, _, _ in steps_using("actions/checkout@"))


@pytest.mark.parametrize(("name", "number", "block"), steps_using("actions/checkout@"))
def test_no_checkout_leaves_its_token_behind(name: str, number: int, block: list[str]) -> None:
    """A persisted token sits in ``.git/config`` for every script and build backend after it."""
    assert setting(block, "persist-credentials") == "false", (
        f"{name}:{number}: a checkout without `persist-credentials: false` leaves its token in "
        ".git/config while the job's fetched code runs; nothing here pushes"
    )
