"""The release-coherence guard, pinned against the two failures it exists to prevent.

`scripts/check_release.py` is repo tooling, not shipped code, so it is loaded by path. The
tests below are written as the history they come from: each names the real defect it forecloses.

The git-tag half of the guard runs in CI (`fetch-depth: 0`), never here — these tests stay
offline and never shell out, so the tag list is passed in.
"""

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "check_release", Path(__file__).resolve().parent.parent / "scripts" / "check_release.py"
)
check_release = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(check_release)


def changelog(*sections: str) -> str:
    """A CHANGELOG with the real file's preamble and whatever sections a test needs."""
    return "# Changelog\n\nAll notable changes are documented here.\n\n" + "\n\n".join(sections)


def section(version: str, body: str = "Something shipped.") -> str:
    return f"## [{version}] - 2026-07-14\n\n{body}"


UNRELEASED_EMPTY = "## [Unreleased]"
CODE = '__version__ = "0.72.0"\n'


class TestTheErasure:
    """v0.43.0 and v0.47.0 were tagged, published, and had no section left (#221, #240).

    Both times a PR renamed the previous release's header to its own new version and prepended
    its narrative — so the published version vanished and its content was credited to a version
    that was never released. The diff looked like an ordinary new section.
    """

    def test_a_tagged_version_with_no_section_fails(self):
        text = changelog(UNRELEASED_EMPTY, section("0.72.0"), section("0.43.1"))
        errors = check_release.check_repo(text, CODE, ["v0.43.0", "v0.72.0"])
        assert any("v0.43.0 is tagged but has no" in e for e in errors)

    def test_the_restored_history_passes(self):
        text = changelog(UNRELEASED_EMPTY, section("0.72.0"), section("0.43.1"), section("0.43.0"))
        assert check_release.check_repo(text, CODE, ["v0.43.0", "v0.72.0"]) == []

    def test_an_untagged_intermediate_version_is_fine(self):
        """A dozen versions have a section and no tag — they roll into the next tag's wheel.

        The guard checks that every *tag* has a section, never that every section has a tag.
        """
        text = changelog(UNRELEASED_EMPTY, section("0.72.0"), section("0.71.0"), section("0.70.0"))
        assert check_release.check_repo(text, CODE, ["v0.70.0", "v0.72.0"]) == []


class TestTheVacuousPass:
    """A check that cannot find its input is a no-op guard that reads green forever.

    CI's shallow default fetches no tags. Without this, the guard would "pass" on an empty tag
    list and never catch anything again — the basecradle-noc#253 failure shape.
    """

    def test_no_tags_is_an_error_not_a_pass(self):
        text = changelog(UNRELEASED_EMPTY, section("0.72.0"))
        errors = check_release.check_repo(text, CODE, [])
        assert any("Refusing to pass vacuously" in e for e in errors)


class TestBumpAndSectionLandTogether:
    def test_a_bump_with_no_section_fails(self):
        text = changelog(UNRELEASED_EMPTY, section("0.71.0"))
        errors = check_release.check_repo(text, CODE, ["v0.71.0"])
        assert any("_version.py says 0.72.0" in e for e in errors)

    def test_sections_must_descend(self):
        text = changelog(UNRELEASED_EMPTY, section("0.72.0"), section("0.9.0"), section("0.70.0"))
        errors = check_release.check_repo(text, CODE, ["v0.72.0"])
        assert any("descending order" in e for e in errors)

    def test_duplicate_sections_fail(self):
        text = changelog(UNRELEASED_EMPTY, section("0.72.0"), section("0.72.0"))
        errors = check_release.check_repo(text, CODE, ["v0.72.0"])
        assert any("more than one" in e for e in errors)


class TestThePreflight:
    """Issue #308: main carried an untagged bump and a headline fix still under [Unreleased]."""

    def test_staged_changes_block_the_tag(self):
        text = changelog(
            "## [Unreleased]\n\n**The compaction proof counted one tool call per step.**",
            section("0.72.0"),
        )
        errors = check_release.check_release(text, CODE, "v0.72.0", ["v0.72.0"])
        assert any("[Unreleased] is not empty" in e for e in errors)

    def test_a_tag_that_does_not_match_the_code_blocks_the_release(self):
        """hatchling reads the version from the code — this tag would publish a 0.72.0 wheel."""
        text = changelog(UNRELEASED_EMPTY, section("0.73.0"), section("0.72.0"))
        errors = check_release.check_release(text, CODE, "v0.73.0", ["v0.72.0", "v0.73.0"])
        assert any("would publish a 0.72.0 wheel" in e for e in errors)

    def test_releasing_an_undocumented_version_blocks(self):
        text = changelog(UNRELEASED_EMPTY, section("0.72.0"))
        code = '__version__ = "0.73.0"\n'
        errors = check_release.check_release(text, code, "v0.73.0", ["v0.72.0"])
        assert any("releasing an undocumented version" in e for e in errors)

    def test_a_coherent_release_passes(self):
        text = changelog(UNRELEASED_EMPTY, section("0.72.0"), section("0.71.0"))
        assert check_release.check_release(text, CODE, "v0.72.0", ["v0.71.0", "v0.72.0"]) == []


class TestTheGuardNeverCrashes:
    """A guard that dies on an input it did not anticipate is worse than the bug it catches.

    `release.yml` fires on every `v*`, and `v1.0.0rc1` is a legal PEP 440 tag. Ordering versions
    with a bare `int()` raised `ValueError` on it — which would not merely fail that release: the
    tag does not go away, so the `changelog` job would then crash on *every later PR*, bricking
    CI repo-wide with the very check meant to protect it.
    """

    def test_a_pre_release_tag_does_not_crash_the_sort(self):
        assert check_release.sort_key("1.0.0rc1") < check_release.sort_key("1.0.0")

    def test_a_pre_release_tag_is_checked_like_any_other(self):
        text = changelog(UNRELEASED_EMPTY, section("1.0.0"), section("1.0.0rc1"))
        code = '__version__ = "1.0.0"\n'
        assert check_release.check_repo(text, code, ["v1.0.0rc1", "v1.0.0"]) == []

    def test_an_unparseable_version_still_sorts_instead_of_raising(self):
        assert check_release.sort_key("not-a-version") == ((), 0, "not-a-version")


class TestTheRealFilesAgree:
    """The guard's own repository must satisfy it — the check is only as true as its subject."""

    def test_this_repo_is_coherent(self):
        text = check_release.CHANGELOG.read_text()
        code = check_release.VERSION_FILE.read_text()
        assert check_release.version(code) == check_release.released(text)[0]
        assert check_release.unreleased_body(text) == ""


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("", ""),
        ("## [Unreleased]", ""),
        ("## [Unreleased]\n\nStaged.\n\n## [0.72.0] - x", "Staged."),
    ],
)
def test_unreleased_body_reads_only_its_own_section(body, expected):
    assert check_release.unreleased_body(body) == expected
