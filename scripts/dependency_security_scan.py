"""The daily dependency security scan's verdict and its routing (issue #816).

Run from ``.github/workflows/dependency-security-scan.yml``::

    python3 scripts/dependency_security_scan.py --open-issues open.json --plan plan.json

Nothing in this repository audited its dependencies on its own: an advisory against a
package in ``uv.lock`` reached us only if Dependabot opened a pull request for it, and
Dependabot does not move a transitive pin. The first reading of ``uv.lock`` found
advisories against twelve packages, two Critical ones among the transitive pins alone
(#817). The capital learned the same lesson on 2026-11-04, when a Rails security release
went unnoticed for 23 hours (basecradle#750). This is the clock: ``osv-scanner`` against
``uv.lock``, every day, against ``main``. Beside it, on the same clock and the same route,
every locked PyPI release's ``yanked`` flag, read from PyPI's own record (issue #1015): a
yanked release carries no advisory, so ``osv-scanner`` is silent on it, and ``uv.lock`` held
``av`` 18.1.0, yanked for a double free, with nothing to say so.

The decisions are a port of the capital's ``lib/dependency_security_scan.rb``, by way of
the Python SDK's port (basecradle-python#317), kept in its shape and its wording so that
all three read the same; this repository owns its own copy. ``DependencySecurityScan``
decides and calls nothing. It takes each scanner's advisories and the repository's open
issues and returns what to do on GitHub -- nothing, a new issue, an update to the one
already open, or closing one -- and the workflow applies it. ``main()`` runs the scanners
and writes that plan as JSON; it never calls GitHub itself.

Standard library only, so the workflow runs it with the runner's own ``python3``.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ADVISORIES_TITLE = "[SECURITY] Dependency Advisories Against main (Daily Scan)"
ERROR_TITLE = "Daily Dependency Security Scan Could Not Reach a Verdict"

# Ruled by @origin, 2026-11-05 (basecradle#750): an advisory rated High or Critical pages
# him; everything else goes to the capital's inbox. An unrated advisory is not known to be
# High, so it does not page; it pages the day it is rated High.
PAGING_SEVERITIES = frozenset({"critical", "high"})
PAGING_LABEL = "needs-human"
INBOX_LABEL = "needs-capital"

SEVERITY_ORDER = ("critical", "high", "medium", "moderate", "low", "none", "unknown")
# A scanner's key is its findings' ecosystem: what a failed scanner's earlier findings are
# carried forward by, so the yanked-release check and osv-scanner never vouch for each other.
YANKED = "pypi-yanked"
ECOSYSTEM_NAMES = {"pypi": "Python package", YANKED: "Yanked Python release"}

# The only authors whose issues are the scan's own. This repository is public, so anyone
# can open an issue with the scan's title and a marker of their choosing; one the scan did
# not write is never read as what it knows, and never edited, labelled or closed by it.
# (The capital's repository is private, so its scan can go by title alone.) `gh issue list`
# names a GitHub App's bot as `app/<slug>`.
#
# Two, because the scan's identity moved (basecradle#870): it runs as this repository's
# own bot, basecradle-harness-ai[bot], and ran as the capital's basecradle-ai[bot] before.
# An issue either one filed is the scan's, so an issue still open from before the move is
# updated and closed by the new author, never duplicated beside it. The capital's entry
# leaves once no open issue here carries the scan's title under the capital's bot.
SCAN_AUTHORS = frozenset({"app/basecradle-harness-ai", "app/basecradle-ai"})

# Each issue body carries the state it was written for, so the next run can tell a
# standing finding (say nothing) from a changed one.
MARKER = re.compile(r"<!-- dependency-security-scan (\{.*\}) -->")

# The scan's one scanner. Its JSON carries each advisory's CVSS score and, for a GitHub
# advisory, GitHub's own rating; pip-audit's carries neither, so it could not route.
OSV_SCANNER = ("osv-scanner", "scan", "source", "--lockfile", "uv.lock", "--format", "json")

# GitHub's rating vocabulary, which is what a CVSS score is translated into.
GHSA_SEVERITIES = frozenset({"critical", "high", "moderate", "low"})


class ScanError(Exception):
    """A scanner ran and did not reach a verdict. Never reads as clean."""


def _key(ecosystem: str, package: str, advisory_id: str) -> str:
    return f"{ecosystem}:{package.replace('@', '')}:{advisory_id}"


def _rank_of(severity: str) -> int:
    return SEVERITY_ORDER.index(severity) if severity in SEVERITY_ORDER else len(SEVERITY_ORDER)


@dataclass(frozen=True)
class Advisory:
    ecosystem: str
    package: str
    affected: str
    id: str
    severity: str
    title: str
    url: str | None
    # Every name osv-scanner grouped this problem under, its key's id included.
    aliases: tuple[str, ...] = ()

    # No "@": the key is written into the issue, and GitHub reads a bare @name as a
    # mention. A Python package name cannot carry one; the key keeps the capital's rule.
    @property
    def key(self) -> str:
        return _key(self.ecosystem, self.package, self.id)

    # The keys this problem could have been filed under on an earlier day: OSV links a
    # GitHub advisory to a PyPA one after the fact, which moves the primary id.
    @property
    def alias_keys(self) -> list[str]:
        return [_key(self.ecosystem, self.package, alias) for alias in self.aliases]

    @property
    def pages(self) -> bool:
        return self.severity in PAGING_SEVERITIES

    @property
    def rank(self) -> tuple[int, str, str, str]:
        return (_rank_of(self.severity), self.ecosystem, self.package, self.key)


@dataclass(frozen=True)
class Plan:
    action: str
    number: int | None
    title: str
    body: str | None
    comment: str | None
    labels: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_osv_scanner(output: str, exit_status: int) -> list[Advisory]:
    """Read ``osv-scanner scan source --format json``.

    Exit 0 means no finding and exit 1 means at least one; anything that does not fit that
    is not a verdict. An unreachable advisory source exits 127 *with* a well-formed, empty
    report, so the exit status is checked against what the report holds, never trusted on
    its own and never ignored.

    osv-scanner groups the records that describe one problem (a GHSA, its PYSEC twin, the
    CVE behind both) and each group is one advisory here.
    """
    try:
        report = json.loads(output)
    except json.JSONDecodeError as error:
        raise ScanError(
            f"osv-scanner produced no readable report (exit {exit_status}): {type(error).__name__}"
        ) from None

    results = report.get("results") if isinstance(report, dict) else None
    if not isinstance(results, list) or not all(isinstance(result, dict) for result in results):
        raise ScanError(f"osv-scanner produced no results list (exit {exit_status})")

    # One problem is one finding. uv forks a package into one version per Python range when
    # it must, and osv-scanner can group the same problem's records differently per version
    # ([GHSA-a, PYSEC-b] against one, [PYSEC-b] against the other), so findings on one
    # package that share any name are merged, never told apart by their primary id alone.
    # A new finding absorbs every one it shares a name with, so the merge is transitive and
    # the findings kept are always pairwise unrelated.
    advisories: list[Advisory] = []
    for result in results:
        for entry in _list_of_dicts(result.get("packages", [])):
            for advisory in _advisories_in(entry):
                for seen in [a for a in advisories if _related(a, advisory)]:
                    advisories.remove(seen)
                    advisory = _merged(seen, advisory)
                advisories.append(advisory)

    if not ((exit_status == 0 and not advisories) or (exit_status == 1 and advisories)):
        raise ScanError(f"osv-scanner exited {exit_status} with {len(advisories)} advisories")

    return advisories


# Every version in the order osv-scanner gave them, the most severe rating, every name, and
# the better of the two primary names with its own title and link.
def _related(one: Advisory, other: Advisory) -> bool:
    same_package = (one.ecosystem, one.package) == (other.ecosystem, other.package)
    return same_package and bool(set(one.aliases) & set(other.aliases))


def _merged(seen: Advisory, other: Advisory) -> Advisory:
    first, second = (seen, other) if _primary_id([seen.id, other.id]) == seen.id else (other, seen)
    versions = seen.affected.split(", ")
    versions += [v for v in other.affected.split(", ") if v not in versions]
    return replace(
        first,
        affected=", ".join(versions),
        severity=min(seen.severity, other.severity, key=_rank_of),
        aliases=tuple(sorted({*seen.aliases, *other.aliases})),
        # A title that is only its name gives way to the other's real one.
        title=second.title if first.title == first.id else first.title,
    )


def _advisories_in(entry: dict[str, Any]) -> list[Advisory]:
    package = entry.get("package")
    groups = entry.get("groups", [])
    records = entry.get("vulnerabilities", [])
    if (
        not isinstance(package, dict)
        or not isinstance(package.get("name"), str)
        or not package["name"]
        or not isinstance(package.get("version"), str)
        or not isinstance(groups, list)
        or not isinstance(records, list)
        or (records and not groups)
    ):
        raise ScanError("osv-scanner printed a package this scan cannot read")

    by_id = {record["id"]: record for record in _list_of_dicts(records) if "id" in record}
    advisories = []
    for group in groups:
        ids = group.get("ids") if isinstance(group, dict) else None
        if not isinstance(ids, list) or not ids or not all(isinstance(i, str) and i for i in ids):
            raise ScanError("osv-scanner printed an advisory group this scan cannot read")

        primary = _primary_id(ids)
        group_records = [by_id[i] for i in ids if i in by_id]
        names = group.get("aliases")
        names = [n for n in names if isinstance(n, str) and n] if isinstance(names, list) else []
        advisories.append(
            Advisory(
                ecosystem="pypi",
                package=package["name"],
                affected=package["version"],
                id=primary,
                severity=_severity_of(group.get("max_severity"), group_records),
                title=_title_of(primary, group_records),
                aliases=tuple(sorted({*ids, *names})),
                url=(
                    f"https://github.com/advisories/{primary}"
                    if primary.startswith("GHSA-")
                    else f"https://osv.dev/vulnerability/{primary}"
                ),
            )
        )
    return advisories


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ScanError("osv-scanner printed a list this scan cannot read")
    return value


# One problem is known by one name, the same every run: its GitHub advisory when it has
# one (what Dependabot, GitHub and most upstream changelogs cite), then its PyPA advisory,
# then whatever sorts first.
def _primary_id(ids: list[str]) -> str:
    for prefix in ("GHSA-", "PYSEC-"):
        named = [i for i in ids if i.startswith(prefix)]
        if named:
            return min(named)
    return min(ids)


# An advisory is as severe as the most severe rating its records carry: the CVSS score
# osv-scanner computes for the group (the highest across its records), or the rating
# GitHub's reviewers gave it. Neither is "unknown", and unknown does not page.
def _severity_of(max_severity: Any, records: list[dict[str, Any]]) -> str:
    ratings = []
    try:
        score = float(max_severity)
    except (TypeError, ValueError):
        score = None
    if score is not None and 0 <= score <= 10:
        ratings.append(_cvss_rating(score))
    for record in records:
        specific = record.get("database_specific")
        rating = specific.get("severity") if isinstance(specific, dict) else None
        if isinstance(rating, str) and rating.lower() in GHSA_SEVERITIES:
            ratings.append(rating.lower())
    return min(ratings, key=_rank_of) if ratings else "unknown"


# CVSS's qualitative scale (the same bands for v3 and v4), in GitHub's words: GitHub calls
# CVSS "Medium" Moderate, and a mixed table would read as two ratings.
def _cvss_rating(score: float) -> str:
    if score >= 9.0:
        return "critical"
    if score >= 7.0:
        return "high"
    if score >= 4.0:
        return "moderate"
    if score > 0:
        return "low"
    return "none"


# A table cell, never a paragraph: a details paragraph is often one long line, and an issue
# body has a size limit GitHub enforces by refusing the write.
TITLE_LIMIT = 200


# Any record's summary before any record's details, the primary record's first.
def _title_of(primary: str, records: list[dict[str, Any]]) -> str:
    ordered = sorted(records, key=lambda record: record.get("id") != primary)
    texts = [record.get("summary") for record in ordered] + [r.get("details") for r in ordered]
    for text in texts:
        line = _first_line(text)
        if line:
            return line if len(line) <= TITLE_LIMIT else line[: TITLE_LIMIT - 1].rstrip() + "…"
    return primary


BOLD_HEADING = re.compile(r"(\*\*|__)[^*_]+\1:?")


# The first line that says something. Details often open with a heading ("### Impact",
# "Impact" over a "------" rule, "**Impact**"), which says nothing.
def _first_line(text: Any) -> str | None:
    lines = [line.strip() for line in text.splitlines()] if isinstance(text, str) else []
    in_fence = False
    for index, line in enumerate(lines):
        following = lines[index + 1] if index + 1 < len(lines) else ""
        # A code block says nothing a title can: its fences and everything between them.
        if line.startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        if (
            in_fence
            or not line
            or line.startswith("#")
            or set(line) <= set("-=*_`~ ")
            or (following and set(following) <= set("-="))
            or BOLD_HEADING.fullmatch(line)
        ):
            continue
        return line
    return None


class DependencySecurityScan:
    def __init__(
        self,
        *,
        advisories: Iterable[Advisory],
        ref: str,
        sha: str,
        run_url: str,
        errors: dict[str, str] | None = None,
        now: datetime | None = None,
    ) -> None:
        """errors: ``{"pypi-yanked": "why that scan reached no verdict"}``, one entry per scanner
        that failed. A failed scanner says nothing about its ecosystem, so its earlier
        findings are carried forward, never reported as resolved."""
        self.advisories = sorted(advisories, key=lambda advisory: advisory.rank)
        self.errors = dict(errors or {})
        self._ref = ref
        self._sha = sha
        self._run_url = run_url
        self._now = now or datetime.now(UTC)

    @property
    def verdict(self) -> str:
        if self.errors:
            return "error"
        return "advisories" if self.advisories else "clean"

    def plans(self, open_issues: list[dict[str, Any]]) -> list[Plan]:
        """open_issues: the repository's open issues, as ``gh issue list --json
        number,title,body,author`` returns them."""
        own = [issue for issue in open_issues if _author_of(issue) in SCAN_AUTHORS]
        candidates = (self._advisories_plan(own), self._error_plan(own))
        return [plan for plan in candidates if plan is not None]

    def _advisories_plan(self, open_issues: list[dict[str, Any]]) -> Plan | None:
        existing = next((i for i in open_issues if i.get("title") == ADVISORIES_TITLE), None)
        state = _state_in(existing.get("body")).get("advisories", {}) if existing else {}
        # A marker edited by hand is read for what it still says, never trusted for more.
        known = {
            key: severity
            for key, severity in (state.items() if isinstance(state, dict) else [])
            if isinstance(severity, str)
        }
        # Filed under another of its names: the same finding, not a new one and a resolved
        # one, which would page again for what is already on the issue. Never a name that
        # is itself reported this run: that one is its own finding.
        # Two names filed apart and linked since are folded into the one now reported.
        reported = {advisory.key for advisory in self.advisories}
        renamed = []
        for advisory in self.advisories:
            for earlier in advisory.alias_keys:
                if earlier != advisory.key and earlier in known and earlier not in reported:
                    # The most severe rating either name was filed at, so what paged under
                    # one name is not news under the other.
                    severity = known.pop(earlier)
                    known[advisory.key] = min(
                        known.get(advisory.key, severity), severity, key=_rank_of
                    )
                    renamed.append((earlier, advisory.key))
        carried = {key: s for key, s in known.items() if key.split(":")[0] in self.errors}
        current = dict(sorted({**carried, **{a.key: a.severity for a in self.advisories}}.items()))

        added = [a for a in self.advisories if a.key not in known]
        rerated = [a for a in self.advisories if a.key in known and known[a.key] != a.severity]
        # Rated High or Critical after it was filed as something lower: that pages too.
        escalated = [a for a in rerated if a.pages and known[a.key] not in PAGING_SEVERITIES]
        resolved = [key for key in known if key not in current]
        if not added and not rerated and not resolved and not renamed:
            return None

        announced = added + escalated
        labels = [_label_for(announced)] if announced else []
        body = self._advisories_body(current, carried)

        if existing:
            return Plan(
                action="update",
                number=existing["number"],
                title=ADVISORIES_TITLE,
                body=body,
                comment=self._change_comment(
                    added, rerated, resolved, renamed, known, current, labels
                ),
                labels=labels,
            )
        return Plan(
            action="create",
            number=None,
            title=ADVISORIES_TITLE,
            body=body,
            comment=None,
            labels=labels,
        )

    # One issue while the scan cannot run, never one a day: filed when a scanner first
    # fails, updated when which scanners fail changes, and closed by the run that reaches a
    # verdict again (that run is the verification).
    def _error_plan(self, open_issues: list[dict[str, Any]]) -> Plan | None:
        existing = next((i for i in open_issues if i.get("title") == ERROR_TITLE), None)
        failing = sorted(self.errors)

        if existing is None:
            if not failing:
                return None
            return Plan(
                action="create",
                number=None,
                title=ERROR_TITLE,
                body=self._error_body(failing),
                comment=None,
                labels=[INBOX_LABEL],
            )
        if not failing:
            return Plan(
                action="close",
                number=existing["number"],
                title=ERROR_TITLE,
                body=None,
                comment=(
                    f"Every scanner reached a verdict on `{self._sha}` (`{self._ref}`), so the "
                    f"clock is running again. Closed by that run: {self._run_url}"
                ),
                labels=[],
            )
        if failing != _state_in(existing.get("body")).get("failing"):
            return Plan(
                action="update",
                number=existing["number"],
                title=ERROR_TITLE,
                body=self._error_body(failing),
                comment=(
                    f"Which scans fail has changed. Now failing: the {_names(failing)}. "
                    f"`{INBOX_LABEL}` applied for the change.\n\n{self._error_list()}\n\n"
                    f"Run: {self._run_url}"
                ),
                labels=[INBOX_LABEL],
            )
        return None

    def _advisories_body(self, current: dict[str, str], carried: dict[str, str]) -> str:
        table = _table(self.advisories) if self.advisories else ""
        return f"""## What Was Found

{self._headline(carried)}

- **Where:** `uv.lock` at `{self._sha}` (`{self._ref}`).
- **When:** scanned {self._when()}, by the scheduled scan and not by a session: {self._run_url}
- **Actor:** none. These are public upstream disclosures, not events on our systems.
- **Class:** a dependency advisory.
- **Contained so far:** nothing. Nobody has read these against the harness yet.

{table}
{self._carried_note(carried)}
## What Happens Next

The fix is the `security-scan-bump` skill: a bump on its own branch to the minimum version that resolves the advisory (`uv lock --upgrade-package <package>==<fixed version>`), then the Critical and High advisories read against the harness on this issue. `uv.lock` pins this repository's CI and development environment, not what a user or the fleet installs, so a lock bump needs no release; an advisory whose fix sits outside a range in `pyproject.toml` (a floor below it, a cap above it) needs that range moved, which ships in a release and reaches the fleet only when the NOC converges it. An advisory that does not apply here is dismissed in `osv-scanner.toml` (`[[IgnoredVulns]]`, with its reason). A yanked release is listed as an advisory of its own, titled with the reason PyPI gives for the yank; it has no dismissal, and the fix is a bump off it. Closing this issue dismisses nothing: while an advisory stands, the next scan files a new issue.

## How This Was Routed

{self._routing()}

{_marker({"advisories": current})}
"""

    def _headline(self, carried: dict[str, str]) -> str:
        if not self.advisories and carried:
            return (
                "This run found no advisory in what it could scan, and could not scan everything."
            )
        if not self.advisories:
            return "The daily scan of `main` finds no advisory now. Close this issue by hand."
        return (
            f"The daily scan of `main` found **{_count_of(self.advisories)}** against what "
            f"`main` resolves: {_tally(self.advisories)}."
        )

    def _routing(self) -> str:
        if not self.advisories:
            return "Nothing is reported now, so this run applied no label."
        if any(a.pages for a in self.advisories):
            return (
                f"`{PAGING_LABEL}` is applied when an advisory rated High or Critical appears, "
                "and it pages `@origin` (his ruling, 2026-11-05, basecradle#750)."
            )
        return (
            f"`{INBOX_LABEL}`: nothing here is rated High or Critical, so nothing paged. Only "
            "High and Critical page `@origin` (his ruling, 2026-11-05, basecradle#750)."
        )

    def _carried_note(self, carried: dict[str, str]) -> str:
        if not carried:
            return ""
        verb = "finding is" if len(carried) == 1 else "findings are"
        return (
            f"\n**Not re-checked this run:** the {_names(self.errors)} could not run, so "
            f"{len(carried)} earlier {verb} carried forward unlisted, neither confirmed nor "
            "cleared.\n"
        )

    def _change_comment(
        self,
        added: list[Advisory],
        rerated: list[Advisory],
        resolved: list[str],
        renamed: list[tuple[str, str]],
        known: dict[str, str],
        current: dict[str, str],
        labels: list[str],
    ) -> str:
        parts = []
        if added:
            parts.append(
                f"**New since the last scan ({_count_of(added)}: {_tally(added)}):**\n\n"
                f"{_table(added)}"
            )
        if rerated:
            changes = ", ".join(
                f"`{a.key}` {str(known[a.key]).capitalize()} → {a.severity.capitalize()}"
                for a in rerated
            )
            parts.append(f"**Re-rated:** {changes}")
        if resolved:
            listed = ", ".join(f"`{key}`" for key in resolved)
            parts.append(f"**No longer reported ({len(resolved)}):** {listed}")
        if renamed:
            names = ", ".join(f"`{earlier}` is now `{key}`" for earlier, key in renamed)
            parts.append(f"**Known by another name now:** {names}")
        if labels:
            parts.append(f"`{labels[0]}` applied for what is new here.")
        if not current:
            parts.append(f"The scan of `main` at `{self._sha}` is clean. Close this issue by hand.")
        parts.append(f"Run: {self._run_url}")
        return "\n\n".join(parts)

    def _error_body(self, failing: list[str]) -> str:
        return f"""## What Happened

The daily dependency security scan of `main` ran and could not reach a verdict, so it says nothing about whether `main` is clean. While this stands, a new advisory goes unseen.

- **Where:** `{self._sha}` (`{self._ref}`), {self._when()}: {self._run_url}

{self._error_list()}

## What Happens Next

Read the run's log, fix what stopped the scan, and re-run the workflow by hand (`Dependency Security Scan`, *Run workflow*). One issue is filed while the scan cannot run, not one a day, and the run that reaches a verdict again closes it.

{_marker({"failing": failing})}
"""

    def _error_list(self) -> str:
        return "\n".join(
            f"- **{ECOSYSTEM_NAMES.get(ecosystem, ecosystem)} scan:** {_cell(reason)}"
            for ecosystem, reason in sorted(self.errors.items())
        )

    def _when(self) -> str:
        return self._now.strftime("%Y-%m-%d %H:%MZ")


def _author_of(issue: dict[str, Any]) -> str | None:
    author = issue.get("author")
    login = author.get("login") if isinstance(author, dict) else None
    return login if isinstance(login, str) else None


def _state_in(body: Any) -> dict[str, Any]:
    # The last marker, not the first: the real one closes the body, and nothing written
    # after it comes from upstream.
    matches = MARKER.findall(body if isinstance(body, str) else "")
    try:
        state = json.loads(matches[-1]) if matches else {}
    except json.JSONDecodeError:
        return {}
    return state if isinstance(state, dict) else {}


def _marker(state: dict[str, Any]) -> str:
    # "<" and ">" escaped inside the JSON (still the same JSON), so nothing in it can close
    # the comment early.
    data = json.dumps(state, separators=(",", ":")).replace("<", "\\u003c").replace(">", "\\u003e")
    return f"<!-- dependency-security-scan {data} -->"


def _label_for(advisories: list[Advisory]) -> str:
    return PAGING_LABEL if any(a.pages for a in advisories) else INBOX_LABEL


def _names(ecosystems: Iterable[str]) -> str:
    return " and ".join(
        f"{ECOSYSTEM_NAMES.get(ecosystem, ecosystem)} scan" for ecosystem in sorted(ecosystems)
    )


def _table(advisories: list[Advisory]) -> str:
    rows = []
    for a in advisories:
        link = f"[{a.id}]({a.url})" if a.url else a.id
        rows.append(
            f"| {a.severity.capitalize()} | {ECOSYSTEM_NAMES[a.ecosystem]} | {_code(a.package)} "
            f"| {_code(a.affected)} | {_cell(link)} | {_cell(a.title)} |"
        )
    header = [
        "| Severity | Ecosystem | Package | Installed or Affected | Advisory | Title |",
        "|---|---|---|---|---|---|",
    ]
    return "\n".join([*header, *rows]) + "\n"


def _count_of(advisories: list[Advisory]) -> str:
    return f"{len(advisories)} {'advisory' if len(advisories) == 1 else 'advisories'}"


def _tally(advisories: list[Advisory]) -> str:
    counts = Counter(a.severity for a in advisories)
    ordered = sorted(counts.items(), key=lambda item: _rank_of(item[0]))
    return ", ".join(f"{count} {severity.capitalize()}" for severity, count in ordered)


# Upstream text, made safe for a table cell: a pipe would split the row, a "<" could open a
# comment that reads as this scan's own marker, and GitHub reads a bare @name as a mention
# of whoever holds that account. Upstream backticks go first: a stray one pairs with the
# span opened around an @name, which leaves that name outside any span and mentions it.
def _cell(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text)).replace("`", "")
    text = text.replace("<", "&lt;").replace("|", "\\|")
    return re.sub(r"@[\w\-/.]+", lambda match: f"`{match.group(0)}`", text)


# The same, inside a code span, where the pipe is still a cell boundary. GitHub reads no
# mention inside a code span, and the "<" is escaped as in a cell: a lockfile's name and
# version never hold one, so it only ever stops a marker.
def _code(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text)).replace("`", "").replace("|", "\\|")
    return f"`{text.replace('<', '&lt;')}`" if text else ""


def scan_pypi(root: Path) -> list[Advisory]:
    """``osv-scanner`` over ``uv.lock``: every package the harness, its extras and its
    development environment resolve, runtime, extras and dev group alike. It queries the OSV API at run time,
    so the advisory source is as fresh as the run."""
    process = subprocess.run(
        OSV_SCANNER, cwd=root, capture_output=True, text=True, timeout=600, check=False
    )
    print(process.stderr, file=sys.stderr)
    return parse_osv_scanner(process.stdout, process.returncode)


# --- yanked releases (issue #1015) ------------------------------------------------------
#
# PyPI yanks a release (PEP 592) when it should not be installed, and says why; a resolver
# then skips it unless it is pinned exactly, which is what a lockfile does. No advisory
# database carries a yank, so this reads PyPI's own record of every release the lock pins.
# Standard library only and reading nothing but ``uv.lock`` and PyPI, so another Python
# repository's scan takes it as it stands.

PYPI_INDEX = "https://pypi.org/simple"
PYPI_RELEASE = "https://pypi.org/pypi/{name}/{version}/json"
# A generic client name: the request says what reads PyPI, never whose repository it is.
USER_AGENT = "dependency-security-scan (yanked-release check)"
READ_ATTEMPTS = 3
READ_TIMEOUT = 30
# The whole check's budget, in seconds. A normal run reads every release in well under a
# minute; this bounds a PyPI that accepts connections and answers slowly, so the check
# reaches "no verdict" and is filed, rather than the job's own timeout killing the run
# before anything is. With osv-scanner's 600 it fits the job's 20 minutes.
YANKED_BUDGET = 300

# uv's source kinds that name no index: the project itself and anything it was told to
# take from somewhere other than an index. Every other kind, a new one included, is no
# verdict, because skipping a kind this check does not know could skip every release.
NO_INDEX = frozenset({"editable", "virtual", "path", "directory", "git", "url"})


def locked_releases(lock: str) -> list[tuple[str, str]]:
    """Every ``(name, version)`` ``uv.lock`` pins from PyPI, each fork's version apart.

    A package from no index (the project itself, a path, a git checkout) is no release
    PyPI could have yanked. One from an index other than PyPI, a source kind uv did not
    write when this was, and a lock that pins nothing from PyPI cannot be read here, so
    each is no verdict, never a pass."""
    try:
        packages = tomllib.loads(lock).get("package")
    except tomllib.TOMLDecodeError as error:
        raise ScanError(f"uv.lock could not be read: {error}") from None
    if not isinstance(packages, list) or not all(isinstance(p, dict) for p in packages):
        raise ScanError("uv.lock holds no package list this check can read")

    releases = set()
    for entry in packages:
        name, version, source = entry.get("name"), entry.get("version"), entry.get("source")
        if not isinstance(name, str) or not name or not isinstance(source, dict):
            raise ScanError("uv.lock holds a package this check cannot read")
        kinds = set(source)
        if kinds <= NO_INDEX and kinds:
            continue
        if kinds != {"registry"}:
            raise ScanError(f"{name} is locked from a source this check cannot read")
        if source["registry"] != PYPI_INDEX:
            raise ScanError(f"{name} is locked from an index this check cannot read")
        if not isinstance(version, str) or not version:
            raise ScanError(f"uv.lock pins {name} from PyPI with no version")
        releases.add((name, version))
    if not releases:
        raise ScanError("uv.lock pins nothing from PyPI")
    return sorted(releases)


def read_release(name: str, version: str, deadline: float) -> dict[str, Any]:
    """PyPI's JSON record of one release, read before ``deadline`` (``time.monotonic()``).

    A transient failure (a 429, a 5xx, the connection) is tried again, a bounded number of
    times; anything else, a read that never succeeds, and one the deadline cuts short is
    no verdict. A record of any release but the one asked for is no verdict either: it
    would answer for the wrong release."""
    url = PYPI_RELEASE.format(
        name=urllib.parse.quote(name, safe=""), version=urllib.parse.quote(version, safe="")
    )
    request = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": USER_AGENT}
    )
    attempt = 1
    while True:
        remaining = _remaining(deadline)
        try:
            with urllib.request.urlopen(request, timeout=min(READ_TIMEOUT, remaining)) as response:
                record = json.load(response)
            info = record.get("info") if isinstance(record, dict) else None
            if not isinstance(info, dict):
                raise ScanError(f"PyPI's record of {name} {version} is not a release record")
            if info.get("version") != version:
                raise ScanError(f"PyPI answered for {name} {version} with another release")
            return record
        except urllib.error.HTTPError as error:
            error.close()
            transient = error.code == 429 or error.code >= 500
            failure = f"PyPI answered {error.code} for {name} {version}"
        # A record that arrived whole and is not JSON: reading it again reads the same.
        except (json.JSONDecodeError, UnicodeDecodeError):
            transient = False
            failure = f"PyPI's record of {name} {version} is not JSON"
        # The connection: DNS, TLS, a reset, a timeout (every one an OSError, URLError
        # included), or a response cut off or malformed on the wire.
        except (OSError, http.client.HTTPException) as error:
            transient = True
            reason = getattr(error, "reason", None) or error
            failure = (
                f"PyPI could not be read for {name} {version}: {type(error).__name__}: {reason}"
            )
        pause = 2**attempt
        if not transient or attempt == READ_ATTEMPTS or _remaining(deadline) <= pause:
            raise ScanError(failure)
        time.sleep(pause)
        attempt += 1


def _remaining(deadline: float) -> float:
    left = deadline - time.monotonic()
    if left <= 0:
        raise ScanError(f"PyPI did not answer for every locked release within {YANKED_BUDGET} s")
    return left


def yanked_release(name: str, version: str, record: dict[str, Any]) -> Advisory | None:
    """A finding when PyPI's record says the release is yanked, titled with PyPI's reason.

    Unrated, because a yank carries no rating: it goes to the capital's inbox, as an
    unrated advisory does, and pages nobody."""
    info = record.get("info")
    yanked = info.get("yanked") if isinstance(info, dict) else None
    if not isinstance(yanked, bool):
        raise ScanError(f"PyPI's record of {name} {version} does not say whether it is yanked")
    if not yanked:
        return None
    # The reason whole, on one line: it is PyPI's words, not a document with headings.
    stated = info.get("yanked_reason")
    reason = " ".join(stated.split()) if isinstance(stated, str) else ""
    if reason and len(reason) > TITLE_LIMIT:
        reason = reason[: TITLE_LIMIT - 1].rstrip() + "…"
    advisory_id = f"yanked-{version}"
    return Advisory(
        ecosystem=YANKED,
        package=name,
        affected=version,
        id=advisory_id,
        severity="unknown",
        title=reason or "Yanked on PyPI, with no reason given",
        url=(
            f"https://pypi.org/project/{urllib.parse.quote(name, safe='')}/"
            f"{urllib.parse.quote(version, safe='')}/"
        ),
        aliases=(advisory_id,),
    )


def scan_yanked(
    root: Path,
    read: Callable[[str, str, float], dict[str, Any]] = read_release,
    budget: float = YANKED_BUDGET,
) -> list[Advisory]:
    """Every release ``uv.lock`` pins from PyPI that PyPI has yanked, all read within
    ``budget`` seconds."""
    try:
        lock = (root / "uv.lock").read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ScanError(f"uv.lock could not be read: {type(error).__name__}") from None
    deadline = time.monotonic() + budget
    findings = []
    for name, version in locked_releases(lock):
        _remaining(deadline)
        finding = yanked_release(name, version, read(name, version, deadline))
        if finding is not None:
            findings.append(finding)
    return findings


SCANNERS: dict[str, Callable[[Path], list[Advisory]]] = {"pypi": scan_pypi, YANKED: scan_yanked}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument(
        "--open-issues",
        required=True,
        type=Path,
        help="Open issues, from `gh issue list --json number,title,body,author`",
    )
    parser.add_argument("--plan", required=True, type=Path, help="Where to write the plan")
    options = parser.parse_args(argv)

    root = Path(__file__).resolve().parent.parent
    advisories: list[Advisory] = []
    errors: dict[str, str] = {}
    for ecosystem, scanner in SCANNERS.items():
        try:
            advisories.extend(scanner(root))
        except ScanError as error:
            errors[ecosystem] = str(error)
        # Whatever stopped a scanner, it reached no verdict, and that is filed, never lost
        # in a red run.
        except Exception as error:  # noqa: BLE001
            errors[ecosystem] = f"{type(error).__name__}: {error}"

    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    repository = os.environ.get("GITHUB_REPOSITORY", "local")
    run_url = f"{server}/{repository}/actions/runs/{os.environ.get('GITHUB_RUN_ID', '0')}"
    scan = DependencySecurityScan(
        advisories=advisories,
        errors=errors,
        ref=os.environ.get("GITHUB_REF_NAME", "local"),
        sha=os.environ.get("GITHUB_SHA", "local"),
        run_url=run_url,
    )
    plans = scan.plans(json.loads(options.open_issues.read_text()))

    plan = {"verdict": scan.verdict, "plans": [p.to_dict() for p in plans]}
    options.plan.write_text(json.dumps(plan, indent=2) + "\n")
    print(
        f"verdict: {scan.verdict} ({len(advisories)} advisories, {len(errors)} scan errors, "
        f"{len(plans)} GitHub actions planned)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
