"""MemPalace's typed collection results, as a fake collection returns them (issue #1026).

MemPalace's ``BaseCollection.get`` and ``.query`` return ``GetResult`` and ``QueryResult``
(``mempalace.backends.base``), read by attribute (``result.ids``). Both still answer the dict style
(``result.get("ids")``, ``result["ids"]``), but only through a shim upstream's own docstring calls
"a migration shim for callers that predate the typed interface", "scheduled for removal": "New code
MUST use attribute access". These carry the same fields and **no** dict protocol, so a harness read
in the old style fails against every fake collection the suite builds, rather than passing until
the release that removes the shim.

`test_palace_results.py` holds them to upstream's field lists when the ``mempalace`` extra is
installed, so a fake cannot drift from the shape it stands in for.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GetResult:
    """``mempalace.backends.base.GetResult``: one flat list per field."""

    ids: list[str]
    documents: list[str]
    metadatas: list[dict]
    embeddings: list[list[float]] | None = None


@dataclass(frozen=True)
class QueryResult:
    """``mempalace.backends.base.QueryResult``: one inner list per query."""

    ids: list[list[str]]
    documents: list[list[str]]
    metadatas: list[list[dict]]
    distances: list[list[float]]
    embeddings: list[list[list[float]]] | None = None


def got(ids, documents=None, metadatas=None, embeddings=None) -> GetResult:
    """A `GetResult` the way MemPalace fills one: a field not asked for is an empty list."""
    ids = list(ids)
    return GetResult(
        ids=ids,
        documents=list(documents) if documents is not None else [],
        metadatas=list(metadatas) if metadatas is not None else [],
        embeddings=None if embeddings is None else list(embeddings),
    )


def queried(ids, distances=None) -> QueryResult:
    """A one-query `QueryResult`: the hit ids, and their distances when asked for."""
    ids = list(ids)
    return QueryResult(
        ids=[ids],
        documents=[[]],
        metadatas=[[]],
        distances=[list(distances) if distances is not None else []],
    )
