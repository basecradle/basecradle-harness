"""``search_params.json`` loader — the operator-owned web-search-parameters source (issue #237).

The sibling of `test_model_params`: this loader turns a JSON object in the config home into the
``parameters`` block of OpenRouter's ``openrouter:web_search`` server tool. These pin its
contract: a missing file is off (``{}`` → the bare tool object, OpenRouter's defaults ride), a
valid object round-trips verbatim (so the full search surface — engine, caps, domain filters,
``user_location`` — is configurable, including keys the harness does not itself name), and
anything malformed is a loud `ValueError` naming the file, failing the wake at startup rather than
searching silently untuned.
"""

import json

import pytest

from basecradle_harness._search_params import SEARCH_PARAMS_NAME, load_search_params


def _write_params(home, content: str) -> None:
    """Write raw ``content`` to ``search_params.json`` under config-home ``home``."""
    (home / SEARCH_PARAMS_NAME).write_text(content, encoding="utf-8")


def test_missing_file_is_empty(tmp_path):
    """No ``search_params.json`` → ``{}`` (the built-in sends the bare object), never an error."""
    assert load_search_params(home=tmp_path) == {}


def test_valid_flat_object(tmp_path):
    """A flat JSON object round-trips verbatim."""
    _write_params(tmp_path, json.dumps({"engine": "exa", "max_results": 10}))
    assert load_search_params(home=tmp_path) == {"engine": "exa", "max_results": 10}


def test_full_surface_including_nested_and_array_values_preserved(tmp_path):
    """The whole documented surface — arrays and a nested ``user_location`` — passes through as-is,
    so an operator can configure everything and a future OpenRouter parameter needs no harness
    change."""
    params = {
        "engine": "exa",
        "max_results": 15,
        "max_total_results": 30,
        "search_context_size": "medium",
        "max_characters": 5000,
        "allowed_domains": ["arxiv.org", "nature.com"],
        "excluded_domains": ["reddit.com"],
        "user_location": {
            "type": "approximate",
            "city": "Dallas",
            "region": "Texas",
            "country": "US",
            "timezone": "America/Chicago",
        },
    }
    _write_params(tmp_path, json.dumps(params))
    assert load_search_params(home=tmp_path) == params


def test_malformed_json_raises_naming_file(tmp_path):
    """Invalid JSON is a hard ``ValueError`` naming the file — a loud fail, not a silent skip."""
    _write_params(tmp_path, "{not valid json")
    with pytest.raises(ValueError, match=SEARCH_PARAMS_NAME):
        load_search_params(home=tmp_path)


@pytest.mark.parametrize("payload", ["[1, 2, 3]", '"a string"', "42", "true", "null"])
def test_non_object_top_level_raises(tmp_path, payload):
    """A top level that is not a JSON object (array/string/number/bool/null) is rejected."""
    _write_params(tmp_path, payload)
    with pytest.raises(ValueError, match=SEARCH_PARAMS_NAME):
        load_search_params(home=tmp_path)


def test_honors_config_home_env(tmp_path, monkeypatch):
    """With no explicit ``home``, the loader resolves ``BASECRADLE_CONFIG_HOME``."""
    monkeypatch.setenv("BASECRADLE_CONFIG_HOME", str(tmp_path))
    _write_params(tmp_path, json.dumps({"engine": "perplexity"}))
    assert load_search_params() == {"engine": "perplexity"}


def test_empty_object(tmp_path):
    """An empty object is valid and yields ``{}`` — the bare tool object."""
    _write_params(tmp_path, "{}")
    assert load_search_params(home=tmp_path) == {}


def test_a_directory_at_the_path_is_a_loud_error(tmp_path):
    """A directory where the file should be is a loud ``ValueError`` naming the path, not a bare
    ``IsADirectoryError`` and not a silent ``{}``."""
    (tmp_path / SEARCH_PARAMS_NAME).mkdir()
    with pytest.raises(ValueError, match=SEARCH_PARAMS_NAME):
        load_search_params(home=tmp_path)
