"""Nothing that makes the published bytes resolves fresh on the day of a tag (issue #916).

Pinning every action to a SHA (issue #914, ``tests/test_workflow_pins.py``) fixes which actions
run, not what they fetch. Whatever builds the wheel is as powerful as whatever uploads it, so each
link of the build is held by ``uv.lock``, where Dependabot moves it as a reviewable diff and the
daily dependency scan reads it:

* **uv** is the ``build`` group's, read off the lock by every ``astral-sh/setup-uv`` step
  (``version-file: uv.lock``). Given no version, setup-uv installs the newest uv; given a
  ``version:`` it installs what the workflow says and the lock no longer decides.
* **The build backend** and everything it imports are exported from the lock as hashed build
  constraints, and ``uv build --require-hashes`` refuses any build requirement without one. Every
  name in ``[build-system].requires`` is declared in the ``build`` group with the same specifier,
  or it would never reach the lock.
* **The interpreter** is uv-managed (``--managed-python``), so it is what the pinned uv knows and
  verifies, never the runner image's.

There is one build, ``.github/actions/build-distributions``, used by the release and by CI alike,
so a bump is built on its pull request before it can build a release. The guards here read the
workflows as text (this repository carries no YAML parser, and adds none for a test); each one is
anchored by a check that it found what it guards, so a reader that matched nothing fails rather
than passes.
"""

from __future__ import annotations

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
FILES = sorted(
    [*(GITHUB / "workflows").glob("*.y*ml"), *(GITHUB / "actions").glob("**/action.y*ml")]
)

PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())
LOCK = tomllib.loads((ROOT / "uv.lock").read_text())

STEP_USES = re.compile(r"^(?P<indent>\s*)(?:-\s+)?uses:\s*(?P<ref>\S+)")


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


def build_group() -> dict[str, Requirement]:
    return {
        Requirement(item).name: Requirement(item)
        for item in PYPROJECT["dependency-groups"]["build"]
    }


def locked(name: str) -> list[dict]:
    return [package for package in LOCK["package"] if package["name"] == name]


# --- uv --------------------------------------------------------------------------------------


def test_the_guard_sees_every_setup_uv() -> None:
    """The build action and CI's lint, test and mempalace jobs: a reader that found none passes."""
    sites = {name for name, _, _ in steps_using("astral-sh/setup-uv@")}
    assert sites == {"actions/build-distributions/action.yml", "workflows/ci.yml"}
    assert len(steps_using("astral-sh/setup-uv@")) >= 4


@pytest.mark.parametrize(("name", "number", "block"), steps_using("astral-sh/setup-uv@"))
def test_every_setup_uv_installs_the_locked_uv(name: str, number: int, block: list[str]) -> None:
    assert setting(block, "version-file") == "uv.lock", (
        f"{name}:{number}: setup-uv without `version-file: uv.lock` installs the newest uv, or the "
        "floor pyproject.toml declares, on the day it runs"
    )
    assert setting(block, "version") is None, (
        f"{name}:{number}: a `version:` input outranks the lock, so CI and the release can disagree"
    )


def test_the_lock_holds_one_exact_uv() -> None:
    """setup-uv takes the first ``uv`` package it finds in the lock, so there must be exactly one."""
    assert "uv" in build_group()
    (package,) = locked("uv")
    assert re.fullmatch(r"\d+\.\d+\.\d+", package["version"])
    assert package["source"] == {"registry": "https://pypi.org/simple"}


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
    text = BUILD_ACTION.read_text()
    assert (
        "uv export --locked --only-group build --no-emit-project --format requirements.txt" in text
    )
    (build,) = [line for line in _commands(text) if line.startswith("uv build")]
    for flag in ("--build-constraints", "--require-hashes", "--managed-python", "--no-cache"):
        assert flag in build, f"the release build lost `{flag}`"
    assert setting(_setup_uv_in(BUILD_ACTION), "enable-cache") == "false", (
        "a restored cache supplies build requirements this run did not download"
    )


def _setup_uv_in(path: Path) -> list[str]:
    lines = path.read_text().splitlines()
    (index,) = [i for i, line in enumerate(lines) if "astral-sh/setup-uv@" in line]
    return step(lines, index)


def _commands(text: str) -> list[str]:
    """Each ``run:`` of a file as one line: a folded ``>-`` block joined with single spaces."""
    commands = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        match = re.match(r"^(?P<indent>\s*)(?:-\s+)?run:\s*(?P<body>.*)$", line)
        if not match:
            continue
        body = match["body"]
        if body in (">-", ">", "|", "|-"):
            indent = len(match["indent"])
            parts = []
            for following in lines[index + 1 :]:
                if following.strip() and len(following) - len(following.lstrip()) <= indent:
                    break
                parts.append(following.strip())
            body = " ".join(part for part in parts if part)
        commands.append(body)
    return commands


def test_nothing_builds_but_the_build_action() -> None:
    """A second ``uv build`` (or ``python -m build``) is a build the constraints do not reach."""
    for path in FILES:
        for command in _commands(path.read_text()):
            if re.search(r"\buv build\b|-m build\b|\bpip wheel\b|\bhatch build\b", command):
                assert path == BUILD_ACTION, f"{path.name} builds outside the build action"


@pytest.mark.parametrize("workflow", [RELEASE, CI])
def test_the_release_and_ci_run_the_one_build(workflow: Path) -> None:
    jobs = re.findall(r"^ {2}build:\n(?:(?: {4,}.*)?\n)+", workflow.read_text(), re.MULTILINE)
    assert len(jobs) == 1, f"{workflow.name} has one `build` job"
    assert "uses: ./.github/actions/build-distributions" in jobs[0]


def test_the_release_job_installs_nothing_of_its_own() -> None:
    """Every tool the release runs is an action pinned to a SHA or a byte the lock holds."""
    installer = re.compile(
        r"\b(?:pip3? install|uv (?:tool |pip )?install|uvx|curl|wget|apt(?:-get)?|npm|npx)\b"
    )
    for path in (RELEASE, BUILD_ACTION):
        for command in _commands(path.read_text()):
            assert not installer.search(command), f"{path.name} installs with `{command}`"


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
