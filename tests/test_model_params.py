"""``model_params.json`` loader — the operator-owned model-parameters source.

The loader turns a JSON object in the config home into the ``**default_params`` the shipped
adapters spread into every model call. These pin its contract: a missing file is off (``{}``),
a valid object round-trips (flat *and* nested), and anything malformed — bad JSON or a
non-object top level — is a loud `ValueError` naming the file, so a typo fails the wake at
startup rather than running silently untuned. The autouse ``_isolated_config_home`` fixture
gives each test its own empty config home; ``BASECRADLE_CONFIG_HOME`` resolution is exercised
by writing into it.
"""

import json

import pytest

from basecradle_harness._model_params import MODEL_PARAMS_NAME, load_model_params


def _write_params(home, content: str) -> None:
    """Write raw ``content`` to ``model_params.json`` under config-home ``home``."""
    (home / MODEL_PARAMS_NAME).write_text(content, encoding="utf-8")


def test_missing_file_is_empty(tmp_path):
    """No ``model_params.json`` → ``{}`` (the feature is simply off), never an error."""
    assert load_model_params(home=tmp_path) == {}


def test_valid_flat_object(tmp_path):
    """A flat JSON object round-trips verbatim."""
    _write_params(tmp_path, json.dumps({"temperature": 0.7, "max_tokens": 4096}))
    assert load_model_params(home=tmp_path) == {"temperature": 0.7, "max_tokens": 4096}


def test_nested_object_preserved(tmp_path):
    """A nested value (e.g. ``reasoning: {effort: high}``) is preserved as-is."""
    _write_params(tmp_path, json.dumps({"reasoning": {"effort": "high"}}))
    assert load_model_params(home=tmp_path) == {"reasoning": {"effort": "high"}}


def test_malformed_json_raises_naming_file(tmp_path):
    """Invalid JSON is a hard ``ValueError`` naming the file — a loud fail, not a silent skip."""
    _write_params(tmp_path, "{not valid json")
    with pytest.raises(ValueError, match=MODEL_PARAMS_NAME):
        load_model_params(home=tmp_path)


@pytest.mark.parametrize("payload", ["[1, 2, 3]", '"a string"', "42", "true", "null"])
def test_non_object_top_level_raises(tmp_path, payload):
    """A top level that is not a JSON object (array/string/number/bool/null) is rejected."""
    _write_params(tmp_path, payload)
    with pytest.raises(ValueError, match=MODEL_PARAMS_NAME):
        load_model_params(home=tmp_path)


def test_honors_config_home_env(tmp_path, monkeypatch):
    """With no explicit ``home``, the loader resolves ``BASECRADLE_CONFIG_HOME``."""
    monkeypatch.setenv("BASECRADLE_CONFIG_HOME", str(tmp_path))
    _write_params(tmp_path, json.dumps({"top_p": 0.9}))
    assert load_model_params() == {"top_p": 0.9}


def test_empty_object(tmp_path):
    """An empty object is valid and yields ``{}``."""
    _write_params(tmp_path, "{}")
    assert load_model_params(home=tmp_path) == {}


def test_a_directory_at_the_path_is_a_loud_error(tmp_path):
    """A directory where the file should be is a loud ``ValueError`` naming the path, not a bare
    ``IsADirectoryError`` and not a silent ``{}``."""
    (tmp_path / MODEL_PARAMS_NAME).mkdir()
    with pytest.raises(ValueError, match=MODEL_PARAMS_NAME):
        load_model_params(home=tmp_path)
