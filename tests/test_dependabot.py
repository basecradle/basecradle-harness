"""The ``publish-path`` Dependabot group names exactly the build toolchain uv.lock holds (issue #994).

The shared auto-merge (``basecradle/.github``) sends an update in a group named ``publish-path``
to the capital instead of merging it. This repository says what that group holds, in
``.github/dependabot.yml``: the ``build`` dependency group and its whole closure in ``uv.lock``,
the packages that make the published bytes (issue #916). A list typed by hand goes stale the day
a toolchain bump brings in a new dependency, and a name it misses merges unread. So the list is
held equal to a walk of the lock: a name the walk finds and the list lacks fails, and so does a
name the list keeps after the lock dropped it.

Both files are read narrowly, failing closed on any shape the reader does not know: this
repository carries no YAML parser, and adds none for a test. The two other parts of the policy
(the seven-day cooldown on every ecosystem, and one ``github-actions`` group of ``"*"``) are
held here too, so a later edit that drops one fails rather than passes.
"""

from __future__ import annotations

import importlib.util
import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DEPENDABOT = ROOT / ".github" / "dependabot.yml"
LOCK = ROOT / "uv.lock"
PROJECT = "basecradle-harness"
BUILD_GROUP = "build"
GROUP = "publish-path"
ECOSYSTEMS = ("uv", "github-actions")

_SPEC = importlib.util.spec_from_file_location(
    "locked_requirement", ROOT / "scripts" / "locked_requirement.py"
)
assert _SPEC is not None and _SPEC.loader is not None
locked_requirement = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(locked_requirement)

#: A pattern Dependabot reads as one exact package name: no wildcard, no quoting games.
EXACT_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def build_closure(lock_text: str) -> set[str]:
    """Every package the project's ``build`` group reaches in the lock, markers included.

    A dependency behind an environment marker is in: whether it installs depends on the
    interpreter, and the list must cover every interpreter the build may use. Each package
    reached must be one PyPI entry with its hashes (``locked_package``), or the walk raises
    ``ValueError``: it cannot tell which of two entries builds.
    """
    lock = tomllib.loads(lock_text)
    projects = [package for package in lock.get("package", []) if package["name"] == PROJECT]
    assert len(projects) == 1, f"uv.lock must hold one `{PROJECT}` to start the walk from"
    start = projects[0].get("dev-dependencies", {}).get(BUILD_GROUP)
    assert start, f"uv.lock's `{PROJECT}` has no `{BUILD_GROUP}` group: nothing to walk"

    walked: set[tuple[str, str | None]] = set()
    pending = list(start)
    while pending:
        dependency = pending.pop()
        name = dependency["name"]
        for extra in [None, *dependency.get("extra", [])]:
            if (name, extra) in walked:
                continue
            walked.add((name, extra))
            package = locked_requirement.locked_package(lock, name)
            if extra is None:
                pending.extend(package.get("dependencies", []))
            else:
                pending.extend(package.get("optional-dependencies", {}).get(extra, []))
    return {name for name, _ in walked}


def entry_lines(text: str, ecosystem: str) -> list[str]:
    """One ecosystem's entry, its blank and comment lines dropped."""
    entries = re.split(r"(?m)^(?=- package-ecosystem:)", text)
    found = [
        e for e in entries if re.match(rf"- package-ecosystem: {ecosystem}\s*$", e, re.MULTILINE)
    ]
    assert len(found) == 1, (
        f"dependabot.yml must have exactly one {ecosystem} entry, not {len(found)}"
    )
    return [
        line for line in found[0].splitlines() if line.strip() and not line.lstrip().startswith("#")
    ]


def group_patterns(text: str, ecosystem: str, group: str) -> list[str]:
    """The ``patterns`` of an ecosystem's first group, which must be ``group``, line by line.

    Fails closed on what would narrow it unseen: another group before it (Dependabot puts a
    package in the first group it matches), any key in the group but ``patterns``
    (``exclude-patterns``; ``dependency-type: production``, since every build package is a dev
    dependency), and an ``allow`` on the entry.
    """
    lines = entry_lines(text, ecosystem)
    narrowing = [line for line in lines if re.match(r"  allow:", line)]
    assert not narrowing, (
        f"the {ecosystem} entry's {narrowing} can drop packages from Dependabot's updates, and "
        f"this reader cannot tell whether it does"
    )
    groups = [i for i, line in enumerate(lines) if line == "  groups:"]
    assert len(groups) == 1 and lines[groups[0] + 1 : groups[0] + 2] == [f"    {group}:"], (
        f"the {ecosystem} entry's `groups:` must open with one `{group}` group, ahead of any other"
    )

    body: list[str] = []
    for line in lines[groups[0] + 2 :]:
        if not line.startswith("      "):
            break
        body.append(line)
    assert body and body[0] == "      patterns:", (
        f"the `{group}` group must open with `patterns:`, and hold nothing else"
    )
    patterns = []
    for line in body[1:]:
        match = re.fullmatch(r"      - (\S+)", line)
        assert match, f"the `{group}` group holds a line its reader cannot vouch for: {line!r}"
        patterns.append(match.group(1))
    return patterns


def ignored_names(text: str, ecosystem: str) -> list[str]:
    """The package names an ecosystem's ``ignore`` list names, read line by line.

    Each item is ``- dependency-name: <name>``, optionally followed by its ``versions`` or
    ``update-types``; any other shape fails, since this reader could not say what it drops.
    """
    lines = entry_lines(text, ecosystem)
    starts = [i for i, line in enumerate(lines) if line == "  ignore:"]
    stray = [line for line in lines if re.match(r"  ignore:", line) and line != "  ignore:"]
    assert len(starts) <= 1 and not stray, (
        f"the {ecosystem} entry's `ignore:` is not one readable list"
    )
    names = []
    for line in lines[starts[0] + 1 :] if starts else []:
        if not line.startswith(("  - ", "    ")):
            break
        if line.startswith("  - "):
            match = re.fullmatch(r"  - dependency-name: (\S+)", line)
            assert match, f"the {ecosystem} `ignore` holds an item it cannot vouch for: {line!r}"
            names.append(match.group(1))
        else:
            assert re.match(r"    (versions|update-types): ", line), (
                f"the {ecosystem} `ignore` holds a line its reader cannot vouch for: {line!r}"
            )
    return names


def check_publish_path_matches_the_lock(dependabot_text: str, lock_text: str) -> None:
    patterns = group_patterns(dependabot_text, "uv", GROUP)
    inexact = [pattern for pattern in patterns if not EXACT_NAME.match(pattern)]
    assert not inexact, (
        f"every `{GROUP}` pattern must be one exact, normalized package name: {inexact}"
    )
    assert len(patterns) == len(set(patterns)), f"`{GROUP}` names a package twice: {patterns}"
    closure = build_closure(lock_text)
    missing = sorted(closure - set(patterns))
    extra = sorted(set(patterns) - closure)
    assert not missing and not extra, (
        f"the `{GROUP}` Dependabot group must name exactly the `{BUILD_GROUP}` group's closure "
        f"in uv.lock, or a bump of a build package merges without the capital reading it."
        f"\n  in the lock, missing from the group: {missing}"
        f"\n  in the group, gone from the lock: {extra}"
    )
    # An ignored build package is one whose bumps nothing would ever propose, read or merge.
    silenced = sorted(set(ignored_names(dependabot_text, "uv")) & closure)
    assert not silenced, (
        f"the uv entry ignores build packages, so nothing would ever move them: {silenced}"
    )


def check_the_cooldown_and_actions_group(dependabot_text: str) -> None:
    for ecosystem in ECOSYSTEMS:
        lines = entry_lines(dependabot_text, ecosystem)
        at = [i for i, line in enumerate(lines) if line == "  cooldown:"]
        assert len(at) == 1 and lines[at[0] + 1 : at[0] + 2] == ["    default-days: 7"], (
            f"the {ecosystem} entry must wait seven days for a new version (`cooldown: "
            f"default-days: 7`), the fleet's policy (issue #994)"
        )
    assert group_patterns(dependabot_text, "github-actions", "github-actions") == ['"*"'], (
        'the github-actions entry must group every action (`"*"`) into one pull request'
    )


def test_the_publish_path_group_names_exactly_the_locked_build_toolchain():
    check_publish_path_matches_the_lock(
        DEPENDABOT.read_text(encoding="utf-8"), LOCK.read_text(encoding="utf-8")
    )


def test_every_ecosystem_waits_seven_days_and_actions_arrive_as_one_group():
    check_the_cooldown_and_actions_group(DEPENDABOT.read_text(encoding="utf-8"))


# --- the readers themselves, on fabricated inputs ---------------------------------------


def _package(name: str, *dependencies: str, **tables: str) -> str:
    deps = ", ".join(f'{{ name = "{dependency}" }}' for dependency in dependencies)
    extra = "".join(f"\n[package.{key}]\n{value}\n" for key, value in tables.items())
    return (
        f'[[package]]\nname = "{name}"\nversion = "1.0.0"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
        f"dependencies = [{deps}]\n"
        f'wheels = [{{ url = "https://example.com/{name}.whl", hash = "sha256:{"b" * 64}" }}]\n'
        f"{extra}\n"
    )


LOCK_TEXT = (
    _package(PROJECT, **{"dev-dependencies": 'build = [{ name = "backend" }]'})
    + _package("backend", "helper")
    + _package("helper")
    + _package("unrelated")
)


def _dependabot(*patterns: str, more: str = "", uv_extra: str = "") -> str:
    items = "".join(f"      - {pattern}\n" for pattern in patterns)
    return (
        "version: 2\nupdates:\n- package-ecosystem: uv\n  directory: /\n"
        "  cooldown:\n    default-days: 7\n"
        f"{uv_extra}  groups:\n    {GROUP}:\n      patterns:\n{items}{more}"
        "  open-pull-requests-limit: 10\n"
        "- package-ecosystem: github-actions\n  directory: /\n"
        "  cooldown:\n    default-days: 7\n  groups:\n"
        '    github-actions:\n      patterns:\n      - "*"\n'
    )


class TestTheReaders:
    def test_accepts_a_group_equal_to_the_closure(self):
        check_publish_path_matches_the_lock(_dependabot("backend", "helper"), LOCK_TEXT)

    def test_accepts_the_policy_shape(self):
        check_the_cooldown_and_actions_group(_dependabot("backend", "helper"))

    def test_walks_past_the_first_level(self):
        assert build_closure(LOCK_TEXT) == {"backend", "helper"}

    def test_follows_an_extra(self):
        lock = (
            _package(PROJECT, **{"dev-dependencies": 'build = [{ name = "backend", extra = ["x"] }]'})
            + _package("backend", **{"optional-dependencies": 'x = [{ name = "plugin" }]'})
            + _package("plugin")
        )  # fmt: skip
        assert build_closure(lock) == {"backend", "plugin"}

    def test_follows_a_cycle_of_extras_to_its_end(self):
        lock = (
            _package(PROJECT, **{"dev-dependencies": 'build = [{ name = "a", extra = ["x"] }]'})
            + _package(
                "a",
                **{"optional-dependencies": 'x = [{ name = "a", extra = ["y"] }]\ny = [{ name = "a", extra = ["x"] }]'},
            )
        )  # fmt: skip
        assert build_closure(lock) == {"a"}

    @pytest.mark.parametrize(
        ("patterns", "reason"),
        [
            (["backend"], "missing from the group: ['helper']"),
            (["backend", "helper", "unrelated"], "gone from the lock: ['unrelated']"),
            (["backend", "helper", "*"], "one exact, normalized package name"),
            (["backend", "helper", "helper"], "names a package twice"),
            (["Backend", "helper"], "one exact, normalized package name"),
        ],
    )
    def test_refuses_a_group_that_differs_from_the_closure(self, patterns, reason):
        with pytest.raises(AssertionError, match=re.escape(reason)):
            check_publish_path_matches_the_lock(_dependabot(*patterns), LOCK_TEXT)

    @pytest.mark.parametrize(
        "more",
        [
            "      exclude-patterns:\n      - helper\n",
            "      dependency-type: production\n",
            "\n      exclude-patterns:\n      - helper\n",
        ],
    )
    def test_refuses_a_key_that_narrows_the_group(self, more):
        with pytest.raises(AssertionError, match="cannot vouch for"):
            group_patterns(_dependabot("backend", "helper", more=more), "uv", GROUP)

    def test_a_package_twice_outside_the_closure_is_left_alone(self):
        assert build_closure(LOCK_TEXT + _package("unrelated")) == {"backend", "helper"}

    def test_refuses_a_package_twice_inside_the_closure(self):
        with pytest.raises(ValueError, match="2 `helper` packages"):
            build_closure(LOCK_TEXT + _package("helper"))

    def test_refuses_a_build_package_the_lock_holds_without_hashes(self):
        lock = LOCK_TEXT.replace(
            _package("helper"), _package("helper").split("wheels = ")[0] + "\n"
        )
        with pytest.raises(ValueError, match="no sha256"):
            build_closure(lock)

    def test_refuses_a_group_ahead_of_it(self):
        text = _dependabot("backend", "helper").replace(
            "  groups:\n", "  groups:\n    everything:\n      patterns:\n      - '*'\n", 1
        )
        with pytest.raises(AssertionError, match="ahead of any other"):
            group_patterns(text, "uv", GROUP)

    def test_refuses_an_allow_that_may_drop_build_packages(self):
        text = _dependabot("backend", "helper", uv_extra="  allow:\n  - dependency-name: helper\n")
        with pytest.raises(AssertionError, match="can drop packages"):
            group_patterns(text, "uv", GROUP)

    def test_a_group_only_in_the_actions_entry_is_not_found(self):
        text = _dependabot("backend").replace(f"    {GROUP}:", "    other:")
        text += f"    {GROUP}:\n      patterns:\n      - backend\n"
        with pytest.raises(AssertionError, match="ahead of any other"):
            group_patterns(text, "uv", GROUP)

    def test_accepts_an_ignore_outside_the_closure(self):
        ignore = '  ignore:\n  - dependency-name: unrelated\n    versions: [">=2"]\n'
        check_publish_path_matches_the_lock(
            _dependabot("backend", "helper", uv_extra=ignore), LOCK_TEXT
        )

    def test_refuses_an_ignore_of_a_build_package(self):
        ignore = "  ignore:\n  - dependency-name: unrelated\n  - dependency-name: helper\n"
        with pytest.raises(AssertionError, match=re.escape("nothing would ever move them")):
            check_publish_path_matches_the_lock(
                _dependabot("backend", "helper", uv_extra=ignore), LOCK_TEXT
            )

    @pytest.mark.parametrize(
        "ignore",
        [
            "  ignore:\n  - {dependency-name: helper}\n",
            "  ignore:\n  - dependency-name: helper\n    when: always\n",
            "  ignore: [{dependency-name: helper}]\n",
            "  ignore:\n  - dependency-name: a\n  ignore:\n  - dependency-name: b\n",
        ],
    )
    def test_refuses_an_ignore_it_cannot_read(self, ignore):
        with pytest.raises(AssertionError, match="cannot vouch for|one readable list"):
            ignored_names(_dependabot("backend", "helper", uv_extra=ignore), "uv")

    @pytest.mark.parametrize(
        ("old", "new"),
        [
            ("  cooldown:\n    default-days: 7\n  groups:\n    publish", "  groups:\n    publish"),
            ("    default-days: 7\n  groups:\n    github", "    default-days: 3\n  groups:\n    github"),
        ],
    )  # fmt: skip
    def test_refuses_an_ecosystem_without_the_seven_day_wait(self, old, new):
        text = _dependabot("backend", "helper").replace(old, new, 1)
        assert text != _dependabot("backend", "helper")
        with pytest.raises(AssertionError, match="seven days"):
            check_the_cooldown_and_actions_group(text)

    def test_refuses_an_actions_group_narrower_than_every_action(self):
        text = _dependabot("backend", "helper").replace('      - "*"\n', "      - actions/*\n")
        with pytest.raises(AssertionError, match="group every action"):
            check_the_cooldown_and_actions_group(text)
