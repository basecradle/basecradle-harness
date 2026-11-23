"""The package's public prices file, read once, for the rate modules that price from it (issue #997).

The numbers the harness prices a call with are **data, and public data**: ``data/prices.json`` in
the package, documented in ``docs/prices.md`` as a stable part of the release, so a product that
prices an account the way the harness does reads the file and never imports a harness module. The
rate modules (`_anthropic_rates`, `_openai_rates`, `_google_rates`) build their tables from it and
keep the arithmetic — usage parsing, tier and geography gating, row choice — as code. One source,
not two: a number lives in the file and nowhere else.

A file this package cannot read is a broken install, so it fails at import like a missing module
would: a file that cannot be read, or is of another format or version, raises `PricesFileError`;
every row is built from exactly the fields its dataclass declares (`row`), so a misspelled or missing
row field is a `PricesFileError` too rather than a default someone has to notice; and a section the
code reads that the file lacks is the ``KeyError`` naming it.
"""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Callable, Mapping
from functools import cache
from importlib import resources
from typing import Any, TypeVar

#: Where the file sits inside the package (``basecradle_harness/data/prices.json`` in the wheel).
RESOURCE = ("data", "prices.json")

#: What the file says it is. A reader checks both before trusting a field.
FORMAT = "basecradle-harness-prices"
FORMAT_VERSION = 1

_Row = TypeVar("_Row")


class PricesFileError(RuntimeError):
    """The package's prices file is missing, unparseable, or not a format this version reads."""


@cache
def load() -> Mapping[str, Any]:
    """The whole prices file, parsed and checked for its format and version."""
    try:
        node = resources.files("basecradle_harness")
        for part in RESOURCE:  # one segment per call: not every 3.11 Traversable takes several
            node = node.joinpath(part)
        data = json.loads(node.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise PricesFileError(f"the package's {'/'.join(RESOURCE)} cannot be read: {exc}") from exc
    if not isinstance(data, dict):
        raise PricesFileError(f"the package's {'/'.join(RESOURCE)} is not a JSON object")
    found = (data.get("format"), data.get("format_version"))
    if found != (FORMAT, FORMAT_VERSION):
        raise PricesFileError(
            f"the package's {'/'.join(RESOURCE)} is {found!r}, not {(FORMAT, FORMAT_VERSION)!r}"
        )
    return data


def vendor(name: str) -> Mapping[str, Any]:
    """One vendor's section of the file."""
    try:
        return load()["vendors"][name]
    except (KeyError, TypeError) as exc:
        raise PricesFileError(f"the prices file carries no {name!r} vendor section") from exc


def source(section: Mapping[str, Any]) -> str:
    """A vendor's ``source`` as one line: the page, any qualifier, and the day it was read."""
    src = section["source"]
    qualifier = f"{src['note']}, " if src["note"] else ""
    return f"{src['url']} ({qualifier}read {src['read_on']})"


def row(cls: type[_Row], fields: Mapping[str, Any]) -> _Row:
    """`cls` built from exactly the fields it declares — no field missing, none extra."""
    declared = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    if set(fields) != declared:
        raise PricesFileError(
            f"a {cls.__name__} row in the prices file has fields {sorted(fields)}, "
            f"not {sorted(declared)}"
        )
    return cls(**fields)


#: What ``long_prompt.when_no_long_row`` may say about a model with no long row whose call is over
#: the threshold: its one row prices the call, or the call is not priced.
NO_LONG_ROW = {"short_row_applies": True, "not_priced": False}


def short_row_without_long(section: Mapping[str, Any]) -> bool:
    """Whether a call over the vendor's threshold, on a model with no long row, is priced on its one row."""
    rule = section["long_prompt"]["when_no_long_row"]
    if rule not in NO_LONG_ROW:
        raise PricesFileError(
            f"long_prompt.when_no_long_row is {rule!r}, not one of {sorted(NO_LONG_ROW)}"
        )
    return NO_LONG_ROW[rule]


def model_key(section: Mapping[str, Any]) -> Callable[[str], str]:
    """The vendor's rule for turning a model id into the key its tables are keyed by.

    In order: surrounding whitespace dropped; the last ``/`` segment taken where
    ``last_path_segment`` is set (a Vertex resource path names the same model as its bare id);
    lower-cased; then a ``strip_suffix`` match removed where one is given (a dated snapshot is
    priced as the id it pins).
    """
    rule = section["model_key"]
    segment = bool(rule["last_path_segment"])
    suffix = re.compile(rule["strip_suffix"]) if rule["strip_suffix"] else None

    def key(model: str) -> str:
        value = model.strip()
        if segment:
            value = value.rsplit("/", 1)[-1]
        value = value.lower()
        return suffix.sub("", value) if suffix is not None else value

    return key
