"""The daily dependency security scan's decisions (issue #816): what it reads, how it routes, and
what it writes on GitHub.

The workflow (``.github/workflows/dependency-security-scan.yml``) only applies the plan
these decisions produce, so this file is where the rules are pinned: one issue and never a
second, High and Critical page and nothing else does, a scan that cannot reach a verdict
never reads as clean, and nothing upstream can break the issue, pose as its marker, or
mention a stranger. Beside ``osv-scanner``, the yanked-release check (issue #1015): a locked
release PyPI has yanked is a finding, on the same route, and a read of PyPI that fails is
never a clean result. A port of the capital's ``test/lib/dependency_security_scan_test.rb``,
test for test, plus what reading ``osv-scanner`` adds.
"""

import importlib.util
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

# `scripts/` is repo tooling, not shipped code, so it is loaded by path, as
# `test_check_release.py` loads its script. Registered in `sys.modules` before it runs:
# `dataclasses` resolves the module's string annotations through it, and 3.10 fails on a
# module it cannot find there.
_SPEC = importlib.util.spec_from_file_location(
    "dependency_security_scan",
    Path(__file__).resolve().parent.parent / "scripts" / "dependency_security_scan.py",
)
assert _SPEC is not None and _SPEC.loader is not None
scan_module = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = scan_module
_SPEC.loader.exec_module(scan_module)

ADVISORIES_TITLE = scan_module.ADVISORIES_TITLE
ERROR_TITLE = scan_module.ERROR_TITLE
Advisory = scan_module.Advisory
DependencySecurityScan = scan_module.DependencySecurityScan
Plan = scan_module.Plan
ScanError = scan_module.ScanError
parse_osv_scanner = scan_module.parse_osv_scanner
YANKED = scan_module.YANKED
REPO_ROOT = Path(__file__).resolve().parent.parent
# What the workflow runs, taken before any test replaces a scanner.
SHIPPED_SCANNERS = dict(scan_module.SCANNERS)


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test reads PyPI: the yanked-release check finds nothing unless a test says
    otherwise, and a read that slips past it fails loudly instead of reaching the network."""
    monkeypatch.setitem(scan_module.SCANNERS, YANKED, lambda root: [])

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a test reached for the network")

    monkeypatch.setattr(scan_module.urllib.request, "urlopen", refuse)


# --- helpers ---------------------------------------------------------------------------


def sole(items: list[Any]) -> Any:
    assert len(items) == 1, items
    return items[0]


def _without_code_spans(text: str) -> str:
    return re.sub(r"`[^`]*`", "", text)


def group(
    *ids: str,
    max_severity: str = "9.3",
    rating: str | None = "CRITICAL",
    summary: str | None = None,
    details: str = "",
    aliases: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {"ids": list(ids), "max_severity": max_severity, "_rating": rating,
            "_summary": summary, "_details": details, "_aliases": list(aliases)}  # fmt: skip


def package(name: str, *groups: dict[str, Any]) -> dict[str, Any]:
    records = []
    for g in groups:
        for record_id in g["ids"]:
            record: dict[str, Any] = {
                "id": record_id,
                "summary": f"Possible problem in {name}"
                if g["_summary"] is None
                else g["_summary"],
                "details": g["_details"],
                "severity": [{"type": "CVSS_V4", "score": "CVSS:4.0/AV:N/AC:L"}],
            }
            if record_id.startswith("GHSA-") and g["_rating"]:
                record["database_specific"] = {"severity": g["_rating"], "github_reviewed": True}
            records.append(record)
    return {
        "package": {"name": name, "version": "4.13.0", "ecosystem": "PyPI"},
        "groups": [{"ids": g["ids"], "aliases": g["ids"] + g["_aliases"],
                    "max_severity": g["max_severity"]} for g in groups],
        "vulnerabilities": records,
    }  # fmt: skip


def report(*packages: dict[str, Any]) -> str:
    results = [
        {"source": {"path": "/repo/uv.lock", "type": "lockfile"}, "packages": list(packages)}
    ]
    return json.dumps({"results": results if packages else [], "experimental_config": {}})


def pypi(
    name: str, id: str, severity: str, title: str | None = None, aliases: tuple[str, ...] = ()
) -> Advisory:
    return Advisory(
        ecosystem="pypi",
        package=name,
        affected="0.28.1",
        id=id,
        severity=severity,
        title=title or f"Possible problem in {name}",
        url=f"https://github.com/advisories/{id}",
        aliases=aliases or (id,),
    )


def context() -> dict[str, Any]:
    return {
        "ref": "main",
        "sha": "0123abc",
        "run_url": "https://github.com/basecradle/basecradle-harness/actions/runs/1",
        "now": datetime(2026, 11, 7, 14, 27, tzinfo=UTC),
    }


def scan_of(*advisories: Advisory, errors: dict[str, str] | None = None) -> DependencySecurityScan:
    return DependencySecurityScan(advisories=advisories, errors=errors or {}, **context())


# How `gh issue list --json author` names this repository's own bot, which the scan runs as
# (basecradle#870), and the capital's, which it ran as before.
SCAN_BOT = {"is_bot": True, "login": "app/basecradle-harness-ai"}
CAPITAL_BOT = {"is_bot": True, "login": "app/basecradle-ai"}
STRANGER = {"id": "U_kgDOC0ffee", "is_bot": False, "login": "john"}


def open_issue(plan: Plan, author: dict[str, Any] = SCAN_BOT) -> dict[str, Any]:
    return {"number": 812, "title": plan.title, "body": plan.body, "author": author}


def unrelated_issue() -> dict[str, Any]:
    return {
        "number": 3,
        "title": "Something Else Entirely",
        "body": '<!-- dependency-security-scan {"keys":["pypi:httpx:GHSA-1111-2222-3333"]} -->',
    }


# --- reading osv-scanner ---------------------------------------------------------------


def test_reads_an_osv_scanner_report_into_advisories() -> None:
    advisory = sole(parse_osv_scanner(report(package("anyio", group("GHSA-1111-2222-3333"))), 1))

    assert advisory.ecosystem == "pypi"
    assert advisory.package == "anyio"
    assert advisory.affected == "4.13.0"
    assert advisory.id == "GHSA-1111-2222-3333"
    assert advisory.severity == "critical"
    assert advisory.url == "https://github.com/advisories/GHSA-1111-2222-3333"
    assert advisory.title == "Possible problem in anyio"
    assert advisory.key == "pypi:anyio:GHSA-1111-2222-3333"
    assert advisory.pages


def test_a_clean_osv_scanner_report_is_no_advisories() -> None:
    assert parse_osv_scanner(report(), 0) == []


def test_one_problem_under_several_names_is_one_advisory_known_by_its_github_name() -> None:
    advisory = sole(
        parse_osv_scanner(
            report(package("certifi", group("PYSEC-2026-4320", "GHSA-4444-5555-6666"))), 1
        )
    )

    assert advisory.id == "GHSA-4444-5555-6666"
    assert advisory.key == "pypi:certifi:GHSA-4444-5555-6666"


def test_an_advisory_with_no_github_name_is_known_by_its_pypa_name_and_linked_on_osv() -> None:
    advisory = sole(
        parse_osv_scanner(report(package("idna", group("PYSEC-2026-0007", "CVE-2026-0007"))), 1)
    )

    assert advisory.id == "PYSEC-2026-0007"
    assert advisory.url == "https://osv.dev/vulnerability/PYSEC-2026-0007"


@pytest.mark.parametrize(
    ("score", "severity"),
    [("9.0", "critical"), ("7.0", "high"), ("6.9", "moderate"), ("4.0", "moderate"),
     ("3.9", "low"), ("0.0", "none")],
)  # fmt: skip
def test_a_cvss_score_is_rated_on_cvss_bands_in_githubs_words(score: str, severity: str) -> None:
    unrated = package("idna", group("PYSEC-2026-0007", max_severity=score, rating=None))

    assert sole(parse_osv_scanner(report(unrated), 1)).severity == severity


def test_an_advisory_is_as_severe_as_its_most_severe_rating() -> None:
    nvd_higher = package(
        "idna", group("GHSA-1111-2222-3333", max_severity="7.5", rating="MODERATE")
    )
    github_higher = package("h11", group("GHSA-4444-5555-6666", max_severity="5.3", rating="HIGH"))

    advisories = parse_osv_scanner(report(nvd_higher, github_higher), 1)

    assert [a.severity for a in advisories] == ["high", "high"]


def test_an_unrated_advisory_is_unknown_and_does_not_page() -> None:
    unrated = package("idna", group("GHSA-1111-2222-3333", max_severity="", rating=None))
    advisory = sole(parse_osv_scanner(report(unrated), 1))

    assert advisory.severity == "unknown"
    assert not advisory.pages


def test_any_records_summary_beats_the_primary_records_details() -> None:
    entry = package("anyio", group("GHSA-1111-2222-3333", "PYSEC-2026-0007"))
    ghsa, pysec = entry["vulnerabilities"]
    ghsa["summary"], ghsa["details"] = "", "### Impact\nA long paragraph."
    pysec["summary"] = "TLS host name spoofing"

    assert sole(parse_osv_scanner(report(entry), 1)).title == "TLS host name spoofing"


@pytest.mark.parametrize(
    "details",
    ["### Impact\n\nFirst", "Impact\n------\nFirst", "Impact\n======\nFirst",
     "**Impact**\nFirst", "__Impact:__\nFirst", "---\nFirst", "```\npip install idna\n```\nFirst",
     "```python\nimport idna\n```\nFirst", "~~~\ncode\n~~~\nFirst"],
)  # fmt: skip
def test_a_details_heading_is_never_the_title(details: str) -> None:
    entry = package("idna", group("GHSA-1111-2222-3333", summary="", details=details))

    assert sole(parse_osv_scanner(report(entry), 1)).title == "First"


def test_a_line_that_merely_opens_and_closes_in_bold_is_still_a_title() -> None:
    entry = package("idna", group("GHSA-1111-2222-3333", summary="**Impact:** crash on **bold**"))

    assert sole(parse_osv_scanner(report(entry), 1)).title == "**Impact:** crash on **bold**"


def test_a_title_is_a_table_cell_never_a_paragraph() -> None:
    entry = package("idna", group("GHSA-1111-2222-3333", summary="word " * 200))

    title = sole(parse_osv_scanner(report(entry), 1)).title

    assert len(title) == scan_module.TITLE_LIMIT
    assert title.endswith("…")


def test_the_title_falls_back_to_the_details_past_any_heading_then_to_the_name() -> None:
    no_summary = package(
        "idna", group("GHSA-1111-2222-3333", summary="", details="### Impact\n\nFirst\nmore")
    )
    nothing = package("h11", group("GHSA-4444-5555-6666", summary="", details=""))

    titles = [a.title for a in parse_osv_scanner(report(no_summary, nothing), 1)]

    assert titles == ["First", "GHSA-4444-5555-6666"]


@pytest.mark.parametrize(
    ("output", "exit_status"),
    [
        ("", 1),
        ("failed to resolve path: stat uv.lock: no such file or directory", 127),
        # An unreachable OSV API exits 127 with a well-formed, empty report.
        (json.dumps({"results": []}), 127),
        # osv-scanner's "no packages found".
        (json.dumps({"results": []}), 128),
        (json.dumps({"results": []}), 1),
        (report(package("anyio", group("GHSA-1111-2222-3333"))), 0),
        (report(package("anyio", group("GHSA-1111-2222-3333"))), 127),
        ("[]", 0),
        ("null", 0),
        ('{"results": ["text"]}', 1),
        ('{"results": [{"packages": [{"package": {"name": "anyio"}}]}]}', 1),
        (('{"results": [{"packages": [{"package": {"name": "a", "version": "1"},'
          ' "vulnerabilities": [{"id": "GHSA-1111-2222-3333"}]}]}]}'), 1),
        # Records with no group to read them by: at exit 0 that would read as clean.
        (('{"results": [{"packages": [{"package": {"name": "a", "version": "1"},'
          ' "vulnerabilities": [{"id": "GHSA-1111-2222-3333"}]}]}]}'), 0),
        (('{"results": [{"packages": [{"package": {"name": "a", "version": "1"},'
          ' "groups": [{"ids": []}]}]}]}'), 1),
    ],
)  # fmt: skip
def test_osv_scanner_output_that_is_not_a_verdict_raises_never_reads_as_clean(
    output: str, exit_status: int
) -> None:
    with pytest.raises(ScanError):
        parse_osv_scanner(output, exit_status)


def test_a_package_forked_into_two_versions_is_one_finding_naming_both() -> None:
    older = package("anyio", group("GHSA-1111-2222-3333"))
    newer = package("anyio", group("GHSA-1111-2222-3333"))
    newer["package"]["version"] = "4.9.0"
    older["package"]["version"] = "4.1.0"

    advisory = sole(parse_osv_scanner(report(older, newer), 1))

    assert advisory.affected == "4.1.0, 4.9.0"


def test_one_problem_grouped_differently_per_fork_is_one_finding() -> None:
    older = package("idna", group("PYSEC-2026-0007", max_severity="7.5", rating=None))
    newer = package("idna", group("GHSA-1111-2222-3333", "PYSEC-2026-0007", rating="LOW"))
    older["package"]["version"] = "3.6"
    newer["package"]["version"] = "3.7"
    newer["vulnerabilities"][0]["summary"] = "The GitHub advisory's own title"

    advisory = sole(parse_osv_scanner(report(older, newer), 1))

    assert advisory.id == "GHSA-1111-2222-3333", "the better name, whichever fork carried it"
    assert advisory.title == "The GitHub advisory's own title"
    assert advisory.url == "https://github.com/advisories/GHSA-1111-2222-3333"
    assert advisory.affected == "3.6, 3.7"
    assert advisory.severity == "critical", "the GHSA record's group scores 9.3"


def test_findings_linked_only_through_a_third_fork_are_still_one_finding() -> None:
    first = package("anyio", group("GHSA-1111-2222-3333"))
    second = package("anyio", group("PYSEC-2026-0007"))
    third = package("anyio", group("GHSA-1111-2222-3333", "PYSEC-2026-0007"))
    for entry, version in ((first, "4.1.0"), (second, "4.2.0"), (third, "4.3.0")):
        entry["package"]["version"] = version

    advisory = sole(parse_osv_scanner(report(first, second, third), 1))

    assert set(advisory.affected.split(", ")) == {"4.1.0", "4.2.0", "4.3.0"}
    assert advisory.id == "GHSA-1111-2222-3333"


def test_a_merged_finding_keeps_a_real_title_over_a_bare_name() -> None:
    nameless = package("idna", group("GHSA-1111-2222-3333", summary="", details=""))
    titled = package("idna", group("PYSEC-2026-0007", "GHSA-1111-2222-3333", summary=""))
    titled["vulnerabilities"][0]["summary"] = "Path traversal in idna"
    titled["package"]["version"] = "3.7"

    assert sole(parse_osv_scanner(report(nameless, titled), 1)).title == "Path traversal in idna"


def test_every_name_of_the_problem_is_kept_as_an_alias() -> None:
    entry = package(
        "certifi",
        group("PYSEC-2026-4320", "GHSA-4444-5555-6666", aliases=("CVE-2026-4320",)),
    )

    advisory = sole(parse_osv_scanner(report(entry), 1))

    assert advisory.aliases == ("CVE-2026-4320", "GHSA-4444-5555-6666", "PYSEC-2026-4320")


def test_a_forked_finding_takes_its_most_severe_rating_and_every_name() -> None:
    older = package("anyio", group("GHSA-1111-2222-3333", max_severity="5.0", rating=None))
    newer = package(
        "anyio",
        group("GHSA-1111-2222-3333", "PYSEC-2026-0007", max_severity="9.8", rating=None),
    )
    older["package"]["version"] = "4.9.0"
    newer["package"]["version"] = "4.10.0"

    advisory = sole(parse_osv_scanner(report(older, newer), 1))

    assert advisory.severity == "critical", "the fork that pages decides"
    assert advisory.affected == "4.9.0, 4.10.0", "in osv-scanner's order, not string order"
    assert advisory.aliases == ("GHSA-1111-2222-3333", "PYSEC-2026-0007")


def test_two_advisories_on_one_package_are_two_findings() -> None:
    entry = package("anyio", group("GHSA-1111-2222-3333"), group("GHSA-4444-5555-6666"))

    keys = [a.key for a in parse_osv_scanner(report(entry), 1)]

    assert keys == ["pypi:anyio:GHSA-1111-2222-3333", "pypi:anyio:GHSA-4444-5555-6666"]


# --- what the scan does on GitHub ------------------------------------------------------


def test_a_clean_scan_with_nothing_open_does_nothing() -> None:
    scan = scan_of()

    assert scan.verdict == "clean"
    assert scan.plans([unrelated_issue()]) == []


def test_a_high_advisory_files_one_security_issue_that_pages() -> None:
    plan = sole(
        scan_of(pypi("httpx", "GHSA-1111-2222-3333", "high"), pypi("h11", "GHSA-4444-5555-6666", "low"))
        .plans([unrelated_issue()])
    )  # fmt: skip

    assert plan.action == "create"
    assert plan.title == ADVISORIES_TITLE
    assert plan.labels == ["needs-human"]
    assert plan.body is not None
    assert "**2 advisories**" in plan.body
    assert "1 High, 1 Low" in plan.body
    assert "0123abc" in plan.body
    assert plan.body.index("GHSA-1111-2222-3333") < plan.body.index("GHSA-4444-5555-6666"), (
        "most severe first"
    )


def test_a_critical_advisory_pages() -> None:
    plan = sole(scan_of(pypi("certifi", "GHSA-1111-2222-3333", "critical")).plans([]))

    assert plan.labels == ["needs-human"]


@pytest.mark.parametrize("severity", ["medium", "moderate", "low", "none", "unknown"])
def test_nothing_below_high_pages_it_goes_to_the_capitals_inbox(severity: str) -> None:
    plan = sole(scan_of(pypi("h11", "GHSA-4444-5555-6666", severity)).plans([]))

    assert plan.labels == ["needs-capital"]


def test_a_standing_finding_is_silent_no_second_issue_no_comment_no_label() -> None:
    scan = scan_of(pypi("httpx", "GHSA-1111-2222-3333", "high"))
    filed = sole(scan.plans([]))

    assert scan.plans([open_issue(filed)]) == []


def test_a_new_high_on_an_open_issue_updates_it_comments_and_pages() -> None:
    filed = sole(scan_of(pypi("h11", "GHSA-4444-5555-6666", "low")).plans([]))
    plan = sole(
        scan_of(pypi("h11", "GHSA-4444-5555-6666", "low"), pypi("httpx", "GHSA-1111-2222-3333", "high"))
        .plans([open_issue(filed)])
    )  # fmt: skip

    assert plan.action == "update"
    assert plan.number == 812
    assert plan.labels == ["needs-human"]
    assert plan.comment is not None and plan.body is not None
    assert "New since the last scan (1 advisory: 1 High)" in plan.comment
    assert "GHSA-1111-2222-3333" in plan.comment
    assert "GHSA-4444-5555-6666" not in plan.comment
    assert "GHSA-4444-5555-6666" in plan.body


def test_a_new_low_on_an_issue_that_already_paged_does_not_page_again() -> None:
    filed = sole(scan_of(pypi("httpx", "GHSA-1111-2222-3333", "high")).plans([]))
    plan = sole(
        scan_of(pypi("httpx", "GHSA-1111-2222-3333", "high"), pypi("h11", "GHSA-4444-5555-6666", "low"))
        .plans([open_issue(filed)])
    )  # fmt: skip

    assert plan.labels == ["needs-capital"]


def test_an_advisory_rated_high_after_it_was_filed_lower_pages_then_and_is_rewritten() -> None:
    filed = sole(scan_of(pypi("httpx", "GHSA-1111-2222-3333", "unknown")).plans([]))
    assert filed.labels == ["needs-capital"]

    plan = sole(
        scan_of(pypi("httpx", "GHSA-1111-2222-3333", "critical")).plans([open_issue(filed)])
    )

    assert plan.action == "update"
    assert plan.labels == ["needs-human"]
    assert plan.comment is not None and plan.body is not None
    assert "**Re-rated:** `pypi:httpx:GHSA-1111-2222-3333` Unknown → Critical" in plan.comment
    assert "| Critical |" in plan.body
    assert (
        scan_of(pypi("httpx", "GHSA-1111-2222-3333", "critical")).plans([open_issue(plan)]) == []
    ), "and is silent again after"


@pytest.mark.parametrize("severity", ["medium", "critical"])
def test_a_rerating_that_does_not_reach_high_rewrites_the_issue_and_labels_nothing(
    severity: str,
) -> None:
    filed = sole(scan_of(pypi("httpx", "GHSA-1111-2222-3333", "high")).plans([]))
    plan = sole(scan_of(pypi("httpx", "GHSA-1111-2222-3333", severity)).plans([open_issue(filed)]))

    assert plan.action == "update"
    assert plan.labels == [], f"High → {severity} is no new page"


def test_an_advisory_that_stops_being_reported_is_noted_and_nothing_is_labeled() -> None:
    filed = sole(
        scan_of(pypi("httpx", "GHSA-1111-2222-3333", "high"), pypi("h11", "GHSA-4444-5555-6666", "low"))
        .plans([])
    )  # fmt: skip
    plan = sole(scan_of(pypi("h11", "GHSA-4444-5555-6666", "low")).plans([open_issue(filed)]))

    assert plan.action == "update"
    assert plan.labels == []
    assert plan.comment is not None
    assert "No longer reported (1):** `pypi:httpx:GHSA-1111-2222-3333`" in plan.comment
    assert "is clean" not in plan.comment


def test_a_scan_that_comes_back_clean_says_so_once_and_never_closes_the_issue() -> None:
    filed = sole(scan_of(pypi("httpx", "GHSA-1111-2222-3333", "high")).plans([]))
    clean = scan_of()
    plan = sole(clean.plans([open_issue(filed)]))

    assert plan.action == "update"
    assert plan.labels == []
    assert plan.comment is not None
    assert "is clean" in plan.comment
    assert "Close this issue by hand" in plan.comment
    assert clean.plans([open_issue(plan)]) == [], "the next clean run is silent"


def test_a_problem_that_gains_a_github_name_is_renamed_on_the_issue_and_never_pages() -> None:
    filed = sole(scan_of(pypi("idna", "PYSEC-2026-0007", "high")).plans([]))
    renamed = pypi(
        "idna", "GHSA-1111-2222-3333", "high", aliases=("GHSA-1111-2222-3333", "PYSEC-2026-0007")
    )

    plan = sole(scan_of(renamed).plans([open_issue(filed)]))

    assert plan.action == "update"
    assert plan.labels == [], "no second page for one problem"
    assert plan.comment is not None and plan.body is not None
    assert (
        "**Known by another name now:** `pypi:idna:PYSEC-2026-0007` is now "
        "`pypi:idna:GHSA-1111-2222-3333`" in plan.comment
    )
    assert "New since" not in plan.comment and "No longer reported" not in plan.comment
    assert '"pypi:idna:GHSA-1111-2222-3333":"high"' in plan.body
    assert scan_of(renamed).plans([open_issue(plan)]) == [], "and silent once renamed"


def test_two_findings_filed_apart_and_linked_since_are_folded_never_reported_resolved() -> None:
    filed = sole(
        scan_of(pypi("idna", "GHSA-1111-2222-3333", "low"), pypi("idna", "PYSEC-2026-0007", "high"))
        .plans([])
    )  # fmt: skip
    linked = pypi(
        "idna", "GHSA-1111-2222-3333", "low", aliases=("GHSA-1111-2222-3333", "PYSEC-2026-0007")
    )

    plan = sole(scan_of(linked).plans([open_issue(filed)]))

    assert plan.labels == []
    assert plan.comment is not None and plan.body is not None
    assert "No longer reported" not in plan.comment
    # The surviving name carries the most severe rating either was filed at, so what paged
    # under one name is never news under the other.
    assert "**Re-rated:** `pypi:idna:GHSA-1111-2222-3333` High → Low" in plan.comment
    assert "`pypi:idna:PYSEC-2026-0007` is now `pypi:idna:GHSA-1111-2222-3333`" in plan.comment
    assert "PYSEC-2026-0007" not in plan.body
    assert scan_of(linked).plans([open_issue(plan)]) == []


def test_a_fold_never_lowers_the_rating_a_finding_already_paged_at() -> None:
    filed = sole(
        scan_of(pypi("idna", "GHSA-1111-2222-3333", "high"), pypi("idna", "PYSEC-2026-0007", "low"))
        .plans([])
    )  # fmt: skip
    linked = pypi(
        "idna", "GHSA-1111-2222-3333", "high", aliases=("GHSA-1111-2222-3333", "PYSEC-2026-0007")
    )

    plan = sole(scan_of(linked).plans([open_issue(filed)]))

    assert plan.labels == [], "it paged as High already"
    assert plan.comment is not None
    assert "Re-rated" not in plan.comment


def test_a_name_reported_as_its_own_finding_is_never_taken_as_an_earlier_name() -> None:
    filed = sole(scan_of(pypi("idna", "GHSA-1111-2222-3333", "low")).plans([]))
    own = pypi("idna", "GHSA-1111-2222-3333", "low")
    linked = pypi(
        "idna", "PYSEC-2026-0007", "low", aliases=("GHSA-1111-2222-3333", "PYSEC-2026-0007")
    )

    plan = sole(scan_of(own, linked).plans([open_issue(filed)]))

    assert plan.comment is not None
    assert "New since the last scan (1 advisory: 1 Low)" in plan.comment
    assert "PYSEC-2026-0007" in plan.comment
    assert "Known by another name" not in plan.comment


def test_a_renamed_finding_that_is_rerated_reads_as_rerated_under_its_new_name() -> None:
    filed = sole(scan_of(pypi("idna", "PYSEC-2026-0007", "low")).plans([]))
    renamed = pypi(
        "idna",
        "GHSA-1111-2222-3333",
        "critical",
        aliases=("GHSA-1111-2222-3333", "PYSEC-2026-0007"),
    )

    plan = sole(scan_of(renamed).plans([open_issue(filed)]))

    assert plan.labels == ["needs-human"]
    assert plan.comment is not None and plan.body is not None
    assert "**Re-rated:** `pypi:idna:GHSA-1111-2222-3333` Low → Critical" in plan.comment
    assert "No longer reported" not in plan.comment
    assert "PYSEC-2026-0007" not in plan.body


def test_an_issue_the_scan_did_not_write_is_never_read_or_touched() -> None:
    advisory = pypi("httpx", "GHSA-1111-2222-3333", "critical")
    # A stranger's issue with the scan's title and a marker saying it already knows.
    forged = open_issue(sole(scan_of(advisory).plans([])), author=STRANGER)
    forged_error = open_issue(sole(scan_of(errors={"pypi": "x"}).plans([])), author=STRANGER)
    forged_error["number"] = 813

    plans = scan_of(advisory).plans([forged, forged_error])

    assert [(p.action, p.number, p.labels) for p in plans] == [("create", None, ["needs-human"])]


@pytest.mark.parametrize("author", [SCAN_BOT, CAPITAL_BOT], ids=["own-bot", "capital-bot"])
def test_an_issue_either_scan_identity_filed_is_the_scans(author: dict[str, Any]) -> None:
    # The scan moved from the capital's bot to this repository's own (basecradle#870): an
    # issue either one filed is updated in place, never duplicated by a fresh one beside it.
    advisory = pypi("httpx", "GHSA-1111-2222-3333", "critical")
    standing = open_issue(sole(scan_of(advisory).plans([])), author=author)

    assert scan_of(advisory).plans([standing]) == []
    assert sole(scan_of().plans([standing])).action == "update"

    # And the could-not-run issue: the run that recovers closes it, whoever filed it.
    stuck = open_issue(sole(scan_of(errors={"pypi": "x"}).plans([])), author=author)
    recovered = sole(scan_of().plans([stuck]))
    assert (recovered.action, recovered.number) == ("close", stuck["number"])


def test_an_issue_a_sibling_bot_filed_is_not_the_scans() -> None:
    advisory = pypi("httpx", "GHSA-1111-2222-3333", "critical")
    sibling = {"is_bot": True, "login": "app/basecradle-python-ai"}
    issue = open_issue(sole(scan_of(advisory).plans([])), author=sibling)

    assert sole(scan_of(advisory).plans([issue])).action == "create"

    # Nor its could-not-run issue: a recovering run leaves it alone rather than closing it.
    stuck = open_issue(sole(scan_of(errors={"pypi": "x"}).plans([])), author=sibling)
    assert scan_of().plans([stuck]) == []


def test_an_issue_with_no_author_is_not_the_scans() -> None:
    advisory = pypi("httpx", "GHSA-1111-2222-3333", "critical")
    issue = open_issue(sole(scan_of(advisory).plans([])))
    del issue["author"]

    assert sole(scan_of(advisory).plans([issue])).action == "create"


def test_an_issue_whose_marker_was_edited_away_is_treated_as_knowing_nothing() -> None:
    rewritten = {
        "number": 812,
        "title": ADVISORIES_TITLE,
        "body": "rewritten by hand",
        "author": SCAN_BOT,
    }
    plan = sole(scan_of(pypi("httpx", "GHSA-1111-2222-3333", "high")).plans([rewritten]))

    assert plan.action == "update"
    assert plan.labels == ["needs-human"]


def test_a_marker_edited_into_nonsense_is_read_for_what_it_still_says() -> None:
    edited = {
        "number": 812,
        "title": ADVISORIES_TITLE,
        "author": SCAN_BOT,
        "body": '<!-- dependency-security-scan {"advisories":'
        '{"pypi:httpx:GHSA-1111-2222-3333":["high"],"pypi:h11:GHSA-4444-5555-6666":"low"}} -->',
    }
    plan = sole(
        scan_of(pypi("httpx", "GHSA-1111-2222-3333", "high"), pypi("h11", "GHSA-4444-5555-6666", "low"))
        .plans([edited])
    )  # fmt: skip

    assert plan.labels == ["needs-human"], "the unreadable entry is new again"
    assert plan.comment is not None
    assert "New since the last scan (1 advisory: 1 High)" in plan.comment


# --- a scan that could not run ---------------------------------------------------------


def test_a_scanner_that_reaches_no_verdict_files_one_issue_for_the_capital() -> None:
    scan = scan_of(errors={"pypi": "osv-scanner exited 127 with 0 advisories"})
    plan = sole(scan.plans([]))

    assert scan.verdict == "error"
    assert plan.action == "create"
    assert plan.title == ERROR_TITLE
    assert plan.labels == ["needs-capital"]
    assert plan.body is not None
    assert "Python package scan:** osv-scanner exited 127" in plan.body
    assert scan.plans([open_issue(plan)]) == [], "one issue while it cannot run, not one a day"


def test_a_second_scanner_failing_is_added_to_the_open_issue_and_the_capital_is_told_again() -> (
    None
):
    filed = sole(scan_of(errors={"pypi": "no verdict"}).plans([]))
    plan = sole(
        scan_of(errors={"pypi": "no verdict", "actions": "no readable report"}).plans(
            [open_issue(filed)]
        )
    )

    assert plan.action == "update"
    assert plan.labels == ["needs-capital"]
    assert plan.comment is not None and plan.body is not None
    assert "the actions scan and Python package scan" in plan.comment
    assert "actions scan:** no readable report" in plan.body


def test_the_run_that_reaches_a_verdict_again_closes_the_could_not_run_issue() -> None:
    filed = sole(scan_of(errors={"pypi": "no verdict"}).plans([]))
    plan = sole(scan_of().plans([open_issue(filed)]))

    assert plan.action == "close"
    assert plan.number == 812
    assert plan.comment is not None
    assert "the clock is running again" in plan.comment


def test_the_scanner_that_did_run_still_reports_beside_the_one_that_could_not() -> None:
    scan = scan_of(pypi("httpx", "GHSA-1111-2222-3333", "high"), errors={"actions": "no verdict"})

    assert scan.verdict == "error"
    assert [(plan.title, plan.labels) for plan in scan.plans([])] == [
        (ADVISORIES_TITLE, ["needs-human"]),
        (ERROR_TITLE, ["needs-capital"]),
    ]


def test_a_failed_scanners_earlier_findings_are_carried_forward_never_reported_resolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    filed = sole(
        scan_of(pypi("httpx", "GHSA-1111-2222-3333", "high"), pypi("h11", "GHSA-4444-5555-6666", "low"))
        .plans([])
    )  # fmt: skip

    scan_down = scan_of(errors={"pypi": "osv-scanner exited 127 with 0 advisories"})
    assert [plan.title for plan in scan_down.plans([open_issue(filed)])] == [ERROR_TITLE]
    # The capital's version has two scanners and so tests one failing beside one finding;
    # this repository has one, so a finding beside a failure comes from another ecosystem.
    other = Advisory("actions", "checkout", "v6", "GHSA-7777-8888-9999", "high", "Bad", None)
    with_other = DependencySecurityScan(
        advisories=[other], errors={"pypi": "no verdict"}, **context()
    )
    monkeypatch.setitem(scan_module.ECOSYSTEM_NAMES, "actions", "GitHub Action")
    plan = with_other.plans([open_issue(filed)])[0]

    assert plan.labels == ["needs-human"]
    assert plan.body is not None and plan.comment is not None
    assert '"pypi:httpx:GHSA-1111-2222-3333":"high"' in plan.body, (
        "the carried finding stays in the marker"
    )
    assert "2 earlier findings are carried forward" in plan.body
    assert "No longer reported" not in plan.comment


# --- what it writes --------------------------------------------------------------------


def test_nothing_it_writes_can_mention_a_github_account() -> None:
    scan = scan_of(
        pypi("httpx", "GHSA-1111-2222-3333", "high", title="Bypass reported by @someone in @org/pkg"),
        errors={"pypi": "failed for @rails/thing"},
    )  # fmt: skip

    for plan in scan.plans([]):
        assert plan.body is not None
        bare = _without_code_spans(plan.body)
        assert "@" not in bare.replace("`@origin`", ""), plan.title


def test_a_stray_backtick_upstream_cannot_free_a_mention() -> None:
    # One upstream backtick would pair with the span opened around "@evil" and leave the
    # name bare. Every backtick left in the cell must be one this scan put around an @name.
    title = "Fix in `parse() reported by @evil and @org/pkg"
    plan = sole(scan_of(pypi("httpx", "GHSA-1111-2222-3333", "high", title=title)).plans([]))

    assert plan.body is not None
    row = next(line for line in plan.body.splitlines() if "GHSA-1111-2222-3333" in line)
    title_cell = row.split(" | ")[-1]
    assert title_cell == "Fix in parse() reported by `@evil` and `@org/pkg` |"
    assert re.sub(r"`@[\w\-/.]+`", "", title_cell).count("`") == 0


def test_a_change_comment_cannot_mention_a_github_account_either() -> None:
    filed = sole(scan_of(pypi("h11", "GHSA-4444-5555-6666", "low")).plans([]))
    plan = sole(
        scan_of(
            pypi("h11", "GHSA-4444-5555-6666", "low"),
            pypi("httpx", "GHSA-1111-2222-3333", "high", title="Found by @someone"),
        ).plans([open_issue(filed)])
    )

    assert plan.comment is not None
    assert "@" not in _without_code_spans(plan.comment)


def test_upstream_text_cannot_break_the_table_or_pose_as_the_marker() -> None:
    hostile = 'Header a | b\nsplits <!-- dependency-security-scan {"advisories":{}} -->'
    scan = scan_of(
        pypi("h11", "GHSA-4444-5555-6666", "low", title=hostile),
        pypi("httpx", "GHSA-1111-2222-3333", "high"),
    )
    plan = sole(scan.plans([]))

    assert plan.body is not None
    assert "Header a \\| b splits &lt;!--" in plan.body
    assert scan.plans([open_issue(plan)]) == [], "the real marker is the one read back"
    rows = [line for line in plan.body.splitlines() if line.startswith(("| High ", "| Low "))]
    assert len(rows) == 2
    for row in rows:
        assert len([c for i, c in enumerate(row) if c == "|" and row[i - 1] != "\\"]) == 7, row


def test_the_real_marker_which_closes_the_body_is_the_one_read() -> None:
    filed = sole(scan_of(pypi("httpx", "GHSA-1111-2222-3333", "high")).plans([]))
    assert filed.body is not None
    # Anything that ever reached the body above the real marker (a hand edit, a quoted
    # earlier body) is not what this scan wrote last, so it is not what it compares with.
    posed = '<!-- dependency-security-scan {"advisories":{}} -->\n\n' + filed.body

    assert scan_module._state_in(posed) == {
        "advisories": {"pypi:httpx:GHSA-1111-2222-3333": "high"}
    }
    assert (
        scan_of(pypi("httpx", "GHSA-1111-2222-3333", "high")).plans(
            [{"number": 812, "title": ADVISORIES_TITLE, "body": posed, "author": SCAN_BOT}]
        )
        == []
    ), "a standing finding stays silent"


def test_the_marker_cannot_close_its_own_comment() -> None:
    marker = scan_module._marker({"advisories": {"pypi:x:-->": "high"}})

    assert marker.count("-->") == 1
    assert marker.endswith(" -->")
    assert scan_module._state_in(marker) == {"advisories": {"pypi:x:-->": "high"}}


# --- yanked releases (issue #1015) -----------------------------------------------------

LOCK = """version = 1
revision = 5
requires-python = ">=3.11"

[[package]]
name = "av"
version = "18.1.0"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "basecradle-harness"
source = { editable = "." }

[[package]]
name = "numpy"
version = "2.3.5"
source = { registry = "https://pypi.org/simple" }
resolution-markers = ["python_full_version >= '3.14'"]

[[package]]
name = "numpy"
version = "2.2.6"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "vendored"
version = "0.1.0"
source = { git = "https://example.com/vendored.git?rev=main#0123abc" }
"""

AV_REASON = (
    "Memory corruption (double free) when decoding audio with custom channel layouts. "
    "Upgrade to 18.1.1"
)


def release_record(yanked: Any, reason: Any = None, version: str = "18.1.0") -> dict[str, Any]:
    return {"info": {"name": "av", "version": version, "yanked": yanked, "yanked_reason": reason}}


def far_off() -> float:
    return scan_module.time.monotonic() + 3600


def test_every_release_the_lock_pins_from_pypi_is_read_each_fork_apart() -> None:
    assert scan_module.locked_releases(LOCK) == [
        ("av", "18.1.0"),
        ("numpy", "2.2.6"),
        ("numpy", "2.3.5"),
    ], "the project itself and a git checkout are no release PyPI could have yanked"


def test_this_repositorys_own_lock_is_one_the_check_can_read() -> None:
    lock = (REPO_ROOT / "uv.lock").read_text()
    releases = scan_module.locked_releases(lock)

    assert len(releases) == lock.count('source = { registry = "https://pypi.org/simple" }')
    assert len(releases) > 100


@pytest.mark.parametrize(
    "lock",
    [
        "this is = not [ toml",
        "version = 1\n",
        '[[package]]\nname = "x"\nversion = "1.0"\nsource = { registry = "https://mirror.example.com/simple" }\n',
        '[[package]]\nname = "x"\nsource = { registry = "https://pypi.org/simple" }\n',
        '[[package]]\nversion = "1.0"\nsource = { registry = "https://pypi.org/simple" }\n',
    ],
    ids=["not-toml", "no-packages", "another-index", "no-version", "no-name"],
)
def test_a_lock_the_check_cannot_read_is_no_verdict_never_a_pass(lock: str) -> None:
    with pytest.raises(ScanError):
        scan_module.locked_releases(lock)


def test_a_yanked_release_is_a_finding_titled_with_pypis_reason() -> None:
    finding = scan_module.yanked_release("av", "18.1.0", release_record(True, AV_REASON))

    assert finding == Advisory(
        ecosystem=YANKED,
        package="av",
        affected="18.1.0",
        id="yanked-18.1.0",
        severity="unknown",
        title=AV_REASON,
        url="https://pypi.org/project/av/18.1.0/",
        aliases=("yanked-18.1.0",),
    )
    assert not finding.pages, "a yank carries no rating, so it goes to the capital's inbox"


def test_a_release_that_is_not_yanked_is_no_finding() -> None:
    assert scan_module.yanked_release("av", "18.1.1", release_record(False)) is None


def test_a_yank_with_no_reason_says_so_and_a_long_one_is_a_table_cell() -> None:
    bare = scan_module.yanked_release("av", "18.1.0", release_record(True, ""))
    long = scan_module.yanked_release("av", "18.1.0", release_record(True, "a\n\n# b " * 200))

    assert bare is not None and bare.title == "Yanked on PyPI, with no reason given"
    assert long is not None and "\n" not in long.title
    assert long.title.startswith("a # b") and long.title.endswith("…")
    assert len(long.title) <= scan_module.TITLE_LIMIT


@pytest.mark.parametrize(
    "record",
    [{}, {"info": None}, {"info": {}}, release_record("true"), release_record(None)],
    ids=["no-info", "null-info", "no-flag", "string-flag", "null-flag"],
)
def test_a_record_that_does_not_say_whether_it_is_yanked_is_no_verdict(
    record: dict[str, Any],
) -> None:
    with pytest.raises(ScanError, match="does not say whether it is yanked"):
        scan_module.yanked_release("av", "18.1.0", record)


def test_the_check_names_each_yanked_release_in_the_lock(tmp_path: Path) -> None:
    (tmp_path / "uv.lock").write_text(LOCK)
    asked = []

    def read(name: str, version: str, deadline: float) -> dict[str, Any]:
        asked.append((name, version))
        return release_record((name, version) == ("av", "18.1.0"), AV_REASON)

    findings = scan_module.scan_yanked(tmp_path, read=read)

    assert asked == [("av", "18.1.0"), ("numpy", "2.2.6"), ("numpy", "2.3.5")]
    assert [(f.key, f.title) for f in findings] == [("pypi-yanked:av:yanked-18.1.0", AV_REASON)]


def test_a_read_that_fails_stops_the_check_it_never_skips_a_release(tmp_path: Path) -> None:
    (tmp_path / "uv.lock").write_text(LOCK)

    def read(name: str, version: str, deadline: float) -> dict[str, Any]:
        if name == "numpy":
            raise ScanError("PyPI answered 404 for numpy 2.2.6")
        return release_record(False)

    with pytest.raises(ScanError, match="404"):
        scan_module.scan_yanked(tmp_path, read=read)


def test_no_lock_is_no_verdict(tmp_path: Path) -> None:
    with pytest.raises(ScanError, match="uv.lock could not be read"):
        scan_module.scan_yanked(tmp_path, read=lambda name, version, deadline: {})


class _Response:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self, *args: Any) -> bytes:
        return self._body

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        return None


def _http_error(code: int) -> Exception:
    return scan_module.urllib.error.HTTPError(
        "https://pypi.org/pypi/av/18.1.0/json", code, "status", None, None
    )


def _pypi_answers(monkeypatch: pytest.MonkeyPatch, *answers: Any) -> list[Any]:
    """PyPI answers each read in turn: an exception is raised, bytes are the body."""
    requests: list[Any] = []
    pending = list(answers)

    def urlopen(request: Any, timeout: float) -> _Response:
        requests.append(request)
        answer = pending.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return _Response(answer)

    monkeypatch.setattr(scan_module.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(scan_module.time, "sleep", lambda seconds: None)
    return requests


def test_the_record_read_is_pypis_own_for_that_exact_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = _pypi_answers(monkeypatch, json.dumps(release_record(True, AV_REASON)).encode())

    assert scan_module.read_release("av", "18.1.0", far_off()) == release_record(True, AV_REASON)
    request = sole(requests)
    assert request.full_url == "https://pypi.org/pypi/av/18.1.0/json"
    assert request.get_header("User-agent") == scan_module.USER_AGENT


@pytest.mark.parametrize(
    "transient",
    [
        _http_error(503),
        _http_error(429),
        scan_module.urllib.error.URLError("reset"),
        TimeoutError(),
        ConnectionResetError(),
        scan_module.http.client.RemoteDisconnected("closed"),
        scan_module.http.client.IncompleteRead(b"{"),
        OSError("tls"),
    ],
    ids=["5xx", "429", "network", "timeout", "reset", "disconnected", "cut-off", "tls"],
)
def test_a_transient_failure_is_read_again(
    monkeypatch: pytest.MonkeyPatch, transient: Exception
) -> None:
    record = release_record(False, version="18.1.1")
    requests = _pypi_answers(monkeypatch, transient, json.dumps(record).encode())

    assert scan_module.read_release("av", "18.1.1", far_off()) == record
    assert len(requests) == 2


def test_a_failure_that_outlasts_the_retries_is_no_verdict(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = _pypi_answers(monkeypatch, *[_http_error(503)] * scan_module.READ_ATTEMPTS)

    with pytest.raises(ScanError, match="PyPI answered 503 for av 18.1.0"):
        scan_module.read_release("av", "18.1.0", far_off())
    assert len(requests) == scan_module.READ_ATTEMPTS


@pytest.mark.parametrize(
    "answer, message",
    [
        (_http_error(404), "answered 404"),
        (b"<html>", "is not JSON"),
        (b"\xff\xfe", "is not JSON"),
        (b"[]", "is not a release record"),
        (b'{"releases": {}}', "is not a release record"),
        (json.dumps(release_record(False, version="18.1.1")).encode(), "with another release"),
    ],
    ids=["404", "not-json", "not-utf8", "not-an-object", "a-project-record", "another-release"],
)
def test_an_answer_that_is_not_a_record_is_no_verdict_and_not_read_again(
    monkeypatch: pytest.MonkeyPatch, answer: Any, message: str
) -> None:
    requests = _pypi_answers(monkeypatch, answer, json.dumps(release_record(False)).encode())

    with pytest.raises(ScanError, match=message):
        scan_module.read_release("av", "18.1.0", far_off())
    assert len(requests) == 1


def test_a_yanked_release_files_on_the_advisory_issue_for_the_capital_and_pages_nobody() -> None:
    finding = scan_module.yanked_release("av", "18.1.0", release_record(True, AV_REASON))
    plan = sole(scan_of(finding).plans([]))

    assert plan.action == "create"
    assert plan.title == ADVISORIES_TITLE
    assert plan.labels == ["needs-capital"]
    assert plan.body is not None
    assert (
        "| Unknown | Yanked Python release | `av` | `18.1.0` "
        "| [yanked-18.1.0](https://pypi.org/project/av/18.1.0/) | Memory corruption" in plan.body
    )


def test_the_yanked_check_failing_carries_only_its_own_findings_forward() -> None:
    yanked = scan_module.yanked_release("av", "18.1.0", release_record(True, AV_REASON))
    advisory = pypi("httpx", "GHSA-1111-2222-3333", "low")
    filed = sole(scan_of(yanked, advisory).plans([]))

    plans = scan_of(errors={YANKED: "PyPI answered 503 for av 18.1.0"}).plans([open_issue(filed)])
    update, error = plans

    assert update.action == "update"
    assert update.comment is not None and update.body is not None
    assert "No longer reported (1):** `pypi:httpx:GHSA-1111-2222-3333`" in update.comment
    assert '"pypi-yanked:av:yanked-18.1.0":"unknown"' in update.body
    assert error.title == ERROR_TITLE
    assert error.body is not None
    assert "Yanked Python release scan:** PyPI answered 503" in error.body


def test_both_scanners_run_on_the_daily_clock() -> None:
    assert SHIPPED_SCANNERS == {"pypi": scan_module.scan_pypi, YANKED: scan_module.scan_yanked}


@pytest.mark.parametrize(
    "source",
    [
        '{ index = "https://pypi.org/simple" }',
        "{ }",
        '{ registry = "https://pypi.org/simple", git = "x" }',
    ],
    ids=["unknown-kind", "empty", "two-kinds"],
)
def test_a_source_kind_the_check_does_not_know_is_no_verdict_never_skipped(source: str) -> None:
    lock = f'[[package]]\nname = "x"\nversion = "1.0"\nsource = {source}\n'

    with pytest.raises(ScanError, match="from a source this check cannot read"):
        scan_module.locked_releases(lock)


@pytest.mark.parametrize("kind", sorted(scan_module.NO_INDEX))
def test_a_package_from_no_index_is_skipped(kind: str) -> None:
    lock = (
        f'[[package]]\nname = "x"\nversion = "1.0"\nsource = {{ {kind} = "." }}\n'
        '[[package]]\nname = "av"\nversion = "18.1.1"\nsource = { registry = "https://pypi.org/simple" }\n'
    )

    assert scan_module.locked_releases(lock) == [("av", "18.1.1")]


def test_a_lock_that_pins_nothing_from_pypi_is_no_verdict() -> None:
    with pytest.raises(ScanError, match="pins nothing from PyPI"):
        scan_module.locked_releases('[[package]]\nname = "x"\nsource = { editable = "." }\n')


def test_the_whole_check_reads_through_the_real_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "uv.lock").write_text(LOCK)
    requests = _pypi_answers(
        monkeypatch,
        json.dumps(release_record(True, AV_REASON)).encode(),
        json.dumps(release_record(False, version="2.2.6")).encode(),
        json.dumps(release_record(False, version="2.3.5")).encode(),
    )

    findings = scan_module.scan_yanked(tmp_path)

    assert [r.full_url for r in requests] == [
        "https://pypi.org/pypi/av/18.1.0/json",
        "https://pypi.org/pypi/numpy/2.2.6/json",
        "https://pypi.org/pypi/numpy/2.3.5/json",
    ]
    assert [f.key for f in findings] == ["pypi-yanked:av:yanked-18.1.0"]


def test_a_check_past_its_budget_is_no_verdict_and_reads_no_more(tmp_path: Path) -> None:
    (tmp_path / "uv.lock").write_text(LOCK)
    asked = []

    def read(name: str, version: str, deadline: float) -> dict[str, Any]:
        asked.append(name)
        return release_record(False, version=version)

    with pytest.raises(ScanError, match="did not answer for every locked release"):
        scan_module.scan_yanked(tmp_path, read=read, budget=-1)
    assert asked == []


def test_a_read_never_waits_past_the_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    timeouts: list[float] = []

    def urlopen(request: Any, timeout: float) -> _Response:
        timeouts.append(timeout)
        raise _http_error(503)

    monkeypatch.setattr(scan_module.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(scan_module.time, "sleep", lambda seconds: pytest.fail("slept"))

    with pytest.raises(ScanError, match="answered 503"):
        scan_module.read_release("av", "18.1.0", scan_module.time.monotonic() + 1.5)
    assert len(timeouts) == 1 and timeouts[0] <= 1.5, "no retry that would outlast the budget"


def test_a_hostile_yank_reason_cannot_mention_break_the_table_or_pose_as_the_marker() -> None:
    reason = "@john | see <!-- dependency-security-scan {} --> `x"
    finding = scan_module.yanked_release("av", "18.1.0", release_record(True, reason))
    plan = sole(scan_of(finding).plans([]))

    assert plan.body is not None
    row = next(line for line in plan.body.splitlines() if "yanked-18.1.0" in line)
    assert row.count(" | ") == 5, "the reason's pipe does not split the row"
    assert "@john" not in _without_code_spans(row)
    assert plan.body.count("<!-- dependency-security-scan") == 1


def test_osv_scanner_failing_carries_only_its_own_findings_forward() -> None:
    yanked = scan_module.yanked_release("av", "18.1.0", release_record(True, AV_REASON))
    advisory = pypi("httpx", "GHSA-1111-2222-3333", "low")
    filed = sole(scan_of(yanked, advisory).plans([]))

    update, _error = scan_of(errors={"pypi": "osv-scanner exited 127 with 0 advisories"}).plans(
        [open_issue(filed)]
    )

    assert update.comment is not None and update.body is not None
    assert "No longer reported (1):** `pypi-yanked:av:yanked-18.1.0`" in update.comment
    assert '"pypi:httpx:GHSA-1111-2222-3333":"low"' in update.body


def test_main_files_a_check_that_raised_something_unforeseen_as_no_verdict(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(scan_module.SCANNERS, "pypi", lambda root: [])

    def unforeseen(root: Any) -> list[Advisory]:
        raise ValueError("something nobody planned for")

    monkeypatch.setitem(scan_module.SCANNERS, YANKED, unforeseen)
    open_issues = tmp_path / "open.json"
    open_issues.write_text("[]")
    plan_file = tmp_path / "plan.json"

    scan_module.main(["--open-issues", str(open_issues), "--plan", str(plan_file)])

    written = json.loads(plan_file.read_text())
    assert written["verdict"] == "error"
    assert (
        "Yanked Python release scan:** ValueError: something nobody planned for"
        in (sole(written["plans"])["body"])
    )


# --- the plan the workflow applies -----------------------------------------------------


def test_main_writes_the_verdict_and_the_plans_the_workflow_applies(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    advisory = pypi("certifi", "GHSA-1111-2222-3333", "critical")
    monkeypatch.setitem(scan_module.SCANNERS, "pypi", lambda root: [advisory])
    open_issues = tmp_path / "open.json"
    open_issues.write_text("[]")
    plan_file = tmp_path / "plan.json"

    assert scan_module.main(["--open-issues", str(open_issues), "--plan", str(plan_file)]) == 0

    written = json.loads(plan_file.read_text())
    assert written["verdict"] == "advisories"
    assert [(p["action"], p["title"], p["labels"]) for p in written["plans"]] == [
        ("create", ADVISORIES_TITLE, ["needs-human"])
    ]
    assert set(written["plans"][0]) == {"action", "number", "title", "body", "comment", "labels"}


def test_main_files_a_scanner_that_crashed_never_loses_it_in_a_red_run(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def crash(root: Any) -> list[Advisory]:
        raise FileNotFoundError("osv-scanner")

    monkeypatch.setitem(scan_module.SCANNERS, "pypi", crash)
    open_issues = tmp_path / "open.json"
    open_issues.write_text("[]")
    plan_file = tmp_path / "plan.json"

    scan_module.main(["--open-issues", str(open_issues), "--plan", str(plan_file)])

    written = json.loads(plan_file.read_text())
    assert written["verdict"] == "error"
    assert [p["title"] for p in written["plans"]] == [ERROR_TITLE]
    assert "FileNotFoundError: osv-scanner" in written["plans"][0]["body"]
