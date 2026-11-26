"""MemPalace's collection results are read by attribute, everywhere (issue #1026).

``mempalace.backends.base`` keeps the dict style on ``GetResult`` and ``QueryResult``
(``result.get("ids")``, ``result["ids"]``) only as "a migration shim for callers that predate the
typed interface", "scheduled for removal in a follow-up cleanup. New code MUST use attribute
access." Every harness read of one used that shim (twenty-two reads), and every fake collection in
the suite returned a plain dict, so nothing could fail until the release that removes it.

Two guards replace that. The fakes return `tests._palace_results`, which has no dict protocol, so
an old-style read fails wherever a fake reaches it. And the scan below fails on an old-style read
anywhere in the package or the suite, reached or not, so a path no fake drives, and a real-palace
test that runs only with the extra, are held too.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

from tests._palace_results import GetResult, QueryResult

#: The typed results' fields (``mempalace.backends.base._TYPED_RESULT_FIELDS``).
FIELDS = frozenset({"ids", "documents", "metadatas", "distances", "embeddings"})

ROOT = Path(__file__).resolve().parent.parent

#: Files the scan skips: the dependency scan reads OSV's JSON, whose ``"ids"`` is no MemPalace
#: result, and this file uses both old forms on purpose, to prove the fakes refuse them.
NOT_A_RESULT = {"tests/test_dependency_security_scan.py", "tests/test_palace_results.py"}


def _old_style_reads(path: Path) -> list[str]:
    """Every ``x.get("<field>")`` and every ``x["<field>"]`` read in one file, as ``line: text``.

    Read from the syntax tree, so a docstring or comment that quotes the old form is not one, and
    a subscript *store* (building an ``upsert``'s keyword arguments) is not a read.
    """
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value in FIELDS
        ) or (
            isinstance(node, ast.Subscript)
            and isinstance(node.ctx, ast.Load)
            and isinstance(node.slice, ast.Constant)
            and node.slice.value in FIELDS
        ):
            found.append(f"{node.lineno}: {ast.unparse(node)}")
    return found


def _sources() -> list[Path]:
    files = sorted((ROOT / "src" / "basecradle_harness").rglob("*.py"))
    files += sorted((ROOT / "tests").rglob("*.py"))
    return [path for path in files if str(path.relative_to(ROOT)) not in NOT_A_RESULT]


def test_no_result_is_read_in_the_dict_style_upstream_will_remove():
    found = {
        str(path.relative_to(ROOT)): reads
        for path in _sources()
        if (reads := _old_style_reads(path))
    }

    assert found == {}, "read a MemPalace result by attribute (result.ids), never result['ids']"


def test_the_scan_finds_both_old_forms(tmp_path):
    """The guard is only as good as its matcher: both forms are found, neither store nor prose."""
    sample = tmp_path / "sample.py"
    sample.write_text(
        '"""result.get("ids") in a docstring is prose."""\n'
        'a = page.get("ids") or []\n'
        'b = found["metadatas"][0]\n'
        'fields["embeddings"] = []\n'
        "c = page.ids\n",
        encoding="utf-8",
    )

    assert _old_style_reads(sample) == ["2: page.get('ids')", "3: found['metadatas']"]


def test_the_fake_results_have_no_dict_protocol():
    result = GetResult(ids=["drawer_1"], documents=["text"], metadatas=[{}])

    with pytest.raises(AttributeError):
        result.get("ids")
    with pytest.raises(TypeError):
        result["ids"]


@pytest.mark.mempalace
@pytest.mark.parametrize("fake", [GetResult, QueryResult])
def test_the_fake_results_carry_upstreams_fields(fake):
    """A fake that drifted from the result it stands in for would prove reads of the wrong shape."""
    base = pytest.importorskip("mempalace.backends.base")
    real = getattr(base, fake.__name__)

    assert [f.name for f in dataclasses.fields(fake)] == [f.name for f in dataclasses.fields(real)]
