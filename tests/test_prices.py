"""The package's public prices file, and the rate tables built from it (issue #997).

The file is the one source of every number the harness prices a call with, and a public part of
the release: a reader outside the package prices from it without importing a harness module. So
these tests pin three things — the tables the rate modules expose are exactly what the file holds
(read here from the raw JSON, never back through the module under test), the file is strict about
its own shape, and every field it carries is documented where its readers look.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import re
import zipfile
from datetime import date
from pathlib import Path

import pytest

from basecradle_harness import _anthropic_rates, _google_rates, _openai_rates, _prices

ROOT = Path(__file__).resolve().parent.parent
FILE = ROOT / "src" / "basecradle_harness" / "data" / "prices.json"
DOC = ROOT / "docs" / "prices.md"
RAW = json.loads(FILE.read_text(encoding="utf-8"))

_SPEC = importlib.util.spec_from_file_location(
    "check_wheel_data", ROOT / "scripts" / "check_wheel_data.py"
)
assert _SPEC is not None and _SPEC.loader is not None
check_wheel_data = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(check_wheel_data)


def _asdict(row):
    return None if row is None else dataclasses.asdict(row)


# --- the file says what it is ----------------------------------------------------------------


def test_the_file_names_its_format_and_version():
    assert RAW["format"] == _prices.FORMAT == "basecradle-harness-prices"
    assert RAW["format_version"] == _prices.FORMAT_VERSION == 1


def test_the_package_reads_the_file_at_its_documented_path():
    assert _prices.RESOURCE == ("data", "prices.json")
    assert _prices.load() == RAW


def test_the_three_vendors_are_carried():
    assert set(RAW["vendors"]) == {"anthropic", "openai", "google"}


# --- the tables are the file ----------------------------------------------------------------


def test_anthropic_tables_are_what_the_file_holds():
    data = RAW["vendors"]["anthropic"]
    assert list(_anthropic_rates.RATES) == list(data["models"])
    for model, rows in data["models"].items():
        rates = _anthropic_rates.RATES[model]
        assert {"short": _asdict(rates.short), "long": _asdict(rates.long)} == {
            "short": rows["short"],
            "long": rows["long"],
        }
    assert _anthropic_rates.HAIKU_LONG_PROMPT_THRESHOLD == data["long_prompt"]["threshold"]
    assert _anthropic_rates.WEB_SEARCH_PER_SEARCH == data["web_search_usd_per_1000"] / 1_000
    assert _anthropic_rates.US_INFERENCE_MULTIPLIER == data["inference_geo_multipliers"]["us"]
    assert _anthropic_rates.PRICED_HOST == data["priced_host"]


def test_openai_tables_are_what_the_file_holds():
    data = RAW["vendors"]["openai"]
    assert list(_openai_rates.RATES) == list(data["models"])
    for model, rows in data["models"].items():
        rates = _openai_rates.RATES[model]
        assert _asdict(rates.short) == rows["short"]
        assert _asdict(rates.long) == rows["long"]
    assert {m: _asdict(r) for m, r in _openai_rates.IMAGE_RATES.items()} == data["image_models"]
    assert {m: _asdict(r) for m, r in _openai_rates.TRANSCRIPTION_RATES.items()} == data[
        "transcription_models"
    ]
    assert _openai_rates.LONG_CONTEXT_THRESHOLD == data["long_prompt"]["threshold"]
    assert _openai_rates.WEB_SEARCH_PER_CALL == data["web_search_usd_per_1000"] / 1_000
    assert _openai_rates.PRICED_HOST == data["priced_host"]


def test_google_tables_are_what_the_file_holds():
    data = RAW["vendors"]["google"]
    assert list(_google_rates.RATES) == list(data["models"])

    def day(value):
        return None if value is None else date.fromisoformat(value)

    def pair(rows):
        return None if rows is None else (rows["short"], rows["long"])

    for model, entry in data["models"].items():
        periods = _google_rates.RATES[model]
        assert len(periods) == len(entry["periods"])
        for rates, period in zip(periods, entry["periods"], strict=True):
            assert (rates.start, rates.end) == (day(period["start"]), day(period["end"]))
            for got, want in (
                (rates.global_, period["global"]),
                (rates.non_global, period["non_global"]),
            ):
                assert (None if got is None else tuple(map(_asdict, got))) == pair(want)
        scheme = entry["grounding"]
        grounding = _google_rates.GROUNDING[model]
        queries, prompts = scheme["usd_per_1000_queries"], scheme["usd_per_1000_prompts"]
        assert grounding.per_query == (None if queries is None else queries / 1_000)
        assert grounding.per_prompt == (None if prompts is None else prompts / 1_000)
        assert grounding.free_tokens is scheme["grounding_tokens_free"]
    assert _google_rates.LONG_CONTEXT_THRESHOLD == data["long_prompt"]["threshold"]
    assert _google_rates.GLOBAL_LOCATION == data["global_location"]


@pytest.mark.parametrize(
    ("module", "line"),
    [
        (
            _anthropic_rates,
            "https://platform.claude.com/docs/en/about-claude/pricing (read 2026-11-21)",
        ),
        (
            _openai_rates,
            "https://developers.openai.com/api/docs/pricing (Standard tier, read 2026-10-07)",
        ),
        (
            _google_rates,
            (
                "https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing "
                "(Standard tier, read 2026-10-06)"
            ),
        ),
    ],
)
def test_each_source_reads_as_the_line_it_always_was(module, line):
    """``SOURCE`` is printed by ``--resolved-config`` and by the fleet's checks; the file's parts
    compose it exactly. When the page moves, this line moves with the file, in the same PR."""
    assert module.SOURCE == line


# --- the rules a reader applies are the rules the harness applies ----------------------------


@pytest.mark.parametrize(
    ("module", "model", "key"),
    [
        (_anthropic_rates, "  Claude-Haiku-4-5-20251001 ", "claude-haiku-4-5"),
        (_anthropic_rates, "claude-opus-5-5", "claude-opus-5-5"),
        (_openai_rates, "GPT-6-Sol-2026-08-14", "gpt-6-sol"),
        (_openai_rates, "gpt-5.4-mini", "gpt-5.4-mini"),
        (
            _google_rates,
            "projects/p/locations/global/publishers/google/models/Gemini-3.8-Flash",
            "gemini-3.8-flash",
        ),
        (_google_rates, "gemini-2.5-pro-20251001", "gemini-2.5-pro-20251001"),
    ],
)
def test_the_model_key_rule_is_the_files(module, model, key):
    assert module._model_key(model) == key


def test_the_no_long_row_rule_is_read_from_the_file():
    """Anthropic prices a flat model on its one row at any length; OpenAI leaves a blank row unpriced."""
    big = _anthropic_rates.HAIKU_LONG_PROMPT_THRESHOLD + 1
    assert _anthropic_rates.call_cost("claude-opus-5-5", _anthropic_rates.Usage(big, 0, 0, 0, 0))
    over = _openai_rates.Usage(input=_openai_rates.LONG_CONTEXT_THRESHOLD + 1, cached=0, output=0)
    assert _openai_rates.call_cost("gpt-5.4-mini", over) is None
    assert _prices.short_row_without_long(
        {"long_prompt": {"when_no_long_row": "short_row_applies"}}
    )
    assert not _prices.short_row_without_long({"long_prompt": {"when_no_long_row": "not_priced"}})


# --- the file is strict about its own shape --------------------------------------------------


def test_a_row_missing_a_field_is_refused():
    row = dict(RAW["vendors"]["anthropic"]["models"]["claude-opus-5-5"]["short"])
    del row["write_1h"]
    with pytest.raises(_prices.PricesFileError, match="write_1h"):
        _prices.row(_anthropic_rates.Tier, row)


def test_a_row_with_an_unknown_field_is_refused():
    row = dict(RAW["vendors"]["openai"]["models"]["gpt-6-sol"]["short"], cache_write_1h=1.0)
    with pytest.raises(_prices.PricesFileError, match="cache_write_1h"):
        _prices.row(_openai_rates.Tier, row)


def test_an_unknown_no_long_row_rule_is_refused():
    with pytest.raises(_prices.PricesFileError, match="when_no_long_row"):
        _prices.short_row_without_long({"long_prompt": {"when_no_long_row": "estimate"}})


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ("{not json", "cannot be read"),
        ("[]", "not a JSON object"),
        (json.dumps({**RAW, "format_version": 2}), "format_version|2"),
        (json.dumps({**RAW, "format": "something-else"}), "something-else"),
    ],
)
def test_a_file_this_version_cannot_read_is_a_loud_error(tmp_path, monkeypatch, content, match):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "prices.json").write_text(content, encoding="utf-8")
    monkeypatch.setattr(_prices.resources, "files", lambda _package: tmp_path)
    _prices.load.cache_clear()
    try:
        with pytest.raises(_prices.PricesFileError, match=match):
            _prices.load()
    finally:
        monkeypatch.undo()
        _prices.load.cache_clear()
    assert _prices.load() == RAW


def test_a_missing_file_is_a_loud_error(tmp_path, monkeypatch):
    monkeypatch.setattr(_prices.resources, "files", lambda _package: tmp_path)
    _prices.load.cache_clear()
    try:
        with pytest.raises(_prices.PricesFileError, match="cannot be read"):
            _prices.load()
    finally:
        monkeypatch.undo()
        _prices.load.cache_clear()


# --- the file is documented where its readers look --------------------------------------------


def _field_names(node, *, under=None):
    """Every field name in the file, skipping the keys that are model ids or geography names."""
    names = set()
    if isinstance(node, dict):
        for key, value in node.items():
            if under not in (
                "models",
                "image_models",
                "transcription_models",
                "vendors",
                "inference_geo_multipliers",
            ):
                names.add(key)
            names |= _field_names(value, under=key)
    elif isinstance(node, list):
        for item in node:
            names |= _field_names(item, under=under)
    return names


def test_every_field_the_file_carries_is_documented():
    doc = DOC.read_text(encoding="utf-8")
    missing = sorted(name for name in _field_names(RAW) if f"`{name}`" not in doc)
    assert not missing, f"docs/prices.md does not document {missing}"


def test_the_doc_names_the_path_and_the_format():
    doc = DOC.read_text(encoding="utf-8")
    assert "basecradle_harness/data/prices.json" in doc
    assert f"`{_prices.FORMAT}`" in doc
    assert re.search(rf"`format_version`\D+{_prices.FORMAT_VERSION}\b", doc)


def test_the_readme_points_at_the_file():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "basecradle_harness/data/prices.json" in readme
    assert "docs/prices.md" in readme


# --- the wheel carries the file (scripts/check_wheel_data.py, run by CI on the built wheel) ----


def _wheel(tmp_path, members):
    wheel = tmp_path / "basecradle_harness-0.0.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return wheel


def test_the_check_publishes_the_prices_file():
    assert "basecradle_harness/data/prices.json" in check_wheel_data.PUBLISHED


def test_a_wheel_carrying_the_file_passes(tmp_path):
    wheel = _wheel(tmp_path, {"basecradle_harness/data/prices.json": FILE.read_bytes()})
    assert check_wheel_data.check(wheel) == []


def test_a_wheel_without_the_file_fails(tmp_path):
    wheel = _wheel(tmp_path, {"basecradle_harness/_prices.py": b""})
    assert check_wheel_data.check(wheel) == [
        f"{wheel.name} carries no basecradle_harness/data/prices.json"
    ]


def test_a_wheel_carrying_other_bytes_fails(tmp_path):
    wheel = _wheel(tmp_path, {"basecradle_harness/data/prices.json": FILE.read_bytes() + b" "})
    (error,) = check_wheel_data.check(wheel)
    assert "differs from src/basecradle_harness/data/prices.json" in error


def test_the_check_wants_exactly_one_wheel(tmp_path, capsys):
    assert check_wheel_data.main([str(tmp_path)]) == 1
    assert "found 0" in capsys.readouterr().err
