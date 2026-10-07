"""Gemini on Vertex through the config layer: the brain factory, the describer's own stack, and
what ``--resolved-config`` says about both (issue #655).

The adapter's own wire behavior is `test_google.py`; this file is about *which* adapter gets built,
from *which* variables — above all, that a describer on Google takes its own key file and never
the brain's, and that a describer on another vendor is never pointed at the brain's endpoint.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from importlib import metadata

import pytest

from basecradle_harness._basecradle import _provider_from_config, resolved_model_params
from basecradle_harness._describer import (
    describer_from_env,
    describer_required_env,
)
from basecradle_harness._google import GoogleProvider
from basecradle_harness._install import config_home
from basecradle_harness._model_params import MODEL_PARAMS_NAME
from basecradle_harness._openrouter import OpenRouterProvider
from basecradle_harness._resolve import resolve_stems
from basecradle_harness._wake import main
from tests.test_google import (  # noqa: F401 - fixtures
    _no_real_google_auth,
    key_file,
    key_info,
    private_key_pem,
)
from tests.test_wake import wake_env  # noqa: F401 - fixture

GEMINI = "gemini-3.8-flash"


@pytest.fixture
def google_brain(monkeypatch, key_file):  # noqa: F811 - fixture
    monkeypatch.setenv("AI_PROVIDER", "google")
    monkeypatch.setenv("AI_SDK", "google-genai")
    monkeypatch.setenv("AI_MODEL", GEMINI)
    monkeypatch.setenv("AI_CREDENTIALS_FILE", str(key_file))
    monkeypatch.setenv("AI_LOCATION", "us")
    monkeypatch.delenv("AI_API_KEY", raising=False)
    return key_file


@pytest.fixture
def openrouter_brain(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "openrouter")
    monkeypatch.setenv("AI_SDK", "openrouter")
    monkeypatch.setenv("AI_MODEL", "z-ai/glm-5.2")
    monkeypatch.setenv("AI_API_KEY", "sk-or-v1-0123456789abcdef0123456789abcdef")
    monkeypatch.setenv("AI_BASE_URL", "https://us.openrouter.ai/api/v1")


@pytest.fixture
def clean_describer(monkeypatch):
    for var in (
        "HARNESS_DESCRIBER_MODEL",
        "HARNESS_DESCRIBER_API_KEY",
        "HARNESS_DESCRIBER_PROVIDERS",
        "HARNESS_DESCRIBER_PROVIDER",
        "HARNESS_DESCRIBER_SDK",
        "HARNESS_DESCRIBER_CREDENTIALS_FILE",
        "HARNESS_DESCRIBER_LOCATION",
        "HARNESS_DESCRIBER_PROJECT",
    ):
        monkeypatch.delenv(var, raising=False)


def _write_model_params(obj):
    path = config_home() / MODEL_PARAMS_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


# === the brain ================================================================================


def test_a_google_brain_is_built_from_its_vertex_variables(google_brain):
    with _provider_from_config("google", "google-genai", "native") as provider:
        assert isinstance(provider, GoogleProvider)
        assert (provider.model, provider.location, provider.project) == (
            GEMINI,
            "us",
            "nova-project",
        )


def test_google_needs_its_own_sdk(google_brain):
    with pytest.raises(ValueError, match="AI_SDK=google-genai"):
        _provider_from_config("google", "openai", "responses")


def test_the_google_sdk_reaches_only_google(google_brain):
    with pytest.raises(ValueError, match="requires AI_PROVIDER=google"):
        _provider_from_config("xai", "google-genai", "native")


def test_harness_owned_keys_never_reach_the_sdk(google_brain, caplog):
    _write_model_params({"temperature": 0.4, "tools": [], "system_instruction": "x"})
    with caplog.at_level(logging.WARNING, logger="basecradle_harness"):
        provider = _provider_from_config("google", "google-genai", "native")
    with provider:
        assert provider.tuning == {"temperature": 0.4}
    assert "'system_instruction'" in caplog.text and "'tools'" in caplog.text


def test_extra_body_is_tuning_on_google_never_stripped(google_brain):
    _write_model_params({"extra_body": {"futureField": 1}, "candidate_count": 2})
    _loaded, stripped = resolved_model_params("google-genai")
    assert stripped == ["candidate_count"]
    with _provider_from_config("google", "google-genai", "native") as provider:
        assert provider.tuning == {"extra_body": {"futureField": 1}}


def test_the_off_box_resolver_knows_google(google_brain):
    resolved = resolve_stems(provider="google", sdk="google-genai")
    assert "memory" in resolved["tools"]


# === the describer ============================================================================


def test_a_describer_on_google_takes_its_own_key_file_never_the_brains(
    openrouter_brain,
    clean_describer,
    monkeypatch,
    tmp_path,
    private_key_pem,  # noqa: F811 - fixture
):
    """The headline case: a GLM brain on OpenRouter, Gemini eyes on Vertex, called direct."""
    own = tmp_path / "describer-key.json"
    own.write_text(json.dumps(key_info(private_key_pem, project_id="nova-eyes")))
    # The brain's Vertex variables, set to values the describer must never pick up.
    monkeypatch.setenv("AI_CREDENTIALS_FILE", str(tmp_path / "never-read.json"))
    monkeypatch.setenv("AI_LOCATION", "eu")
    monkeypatch.setenv("AI_PROJECT", "nova-brain")
    monkeypatch.setenv("HARNESS_DESCRIBER_MODEL", GEMINI)
    monkeypatch.setenv("HARNESS_DESCRIBER_PROVIDER", "google")
    monkeypatch.setenv("HARNESS_DESCRIBER_SDK", "google-genai")
    monkeypatch.setenv("HARNESS_DESCRIBER_CREDENTIALS_FILE", str(own))
    monkeypatch.setenv("HARNESS_DESCRIBER_LOCATION", "us")

    describer = describer_from_env()

    assert describer.fault is None
    adapter = describer.provider
    assert isinstance(adapter, GoogleProvider)
    assert (adapter.model, adapter.location, adapter.project) == (GEMINI, "us", "nova-eyes")
    # The describer's own output budget, and nothing of the brain's tuning.
    assert adapter.tuning == {"max_output_tokens": 2048}
    # A Gemini describer watches a clip natively rather than reading frames.
    assert adapter.supports_video() is True


def test_a_describer_on_another_vendor_never_inherits_the_brains_endpoint(
    monkeypatch, clean_describer, google_brain
):
    """AI_BASE_URL names the brain's vendor's host; a describer on OpenRouter must not be sent there."""
    monkeypatch.setenv("AI_BASE_URL", "https://vertex.proxy.test/")
    monkeypatch.setenv("HARNESS_DESCRIBER_MODEL", "google/gemini-3.8-flash")
    monkeypatch.setenv("HARNESS_DESCRIBER_PROVIDER", "openrouter")
    monkeypatch.setenv("HARNESS_DESCRIBER_SDK", "openrouter")
    monkeypatch.setenv("HARNESS_DESCRIBER_API_KEY", "sk-or-v1-describer0123456789abcdef")
    monkeypatch.setenv("HARNESS_DESCRIBER_PROVIDERS", "google-vertex")

    describer = describer_from_env()

    assert isinstance(describer.provider, OpenRouterProvider)
    assert describer.provider.base_url == "https://openrouter.ai/api/v1"
    describer.provider.close()


@pytest.mark.parametrize(("location", "inherits"), [("us", False), ("US-CENTRAL1", True)])
def test_a_vertex_describer_takes_the_brains_host_only_at_the_brains_location(
    google_brain,
    clean_describer,
    monkeypatch,
    tmp_path,
    private_key_pem,  # noqa: F811 - fixture
    location,
    inherits,
):
    """On Vertex a host is a location: the brain's us-central1 host must not carry a ``us`` call."""
    own = tmp_path / "eyes.json"
    own.write_text(json.dumps(key_info(private_key_pem)))
    host = "https://us-central1-aiplatform.googleapis.com/"
    monkeypatch.setenv("AI_LOCATION", "us-central1")
    monkeypatch.setenv("AI_BASE_URL", host)
    monkeypatch.setenv("HARNESS_DESCRIBER_MODEL", GEMINI)
    monkeypatch.setenv("HARNESS_DESCRIBER_CREDENTIALS_FILE", str(own))
    monkeypatch.setenv("HARNESS_DESCRIBER_LOCATION", location)
    adapter = describer_from_env().provider
    assert adapter.base_url == (host if inherits else None)
    adapter.close()


def test_a_google_brain_describer_still_needs_its_own_key_file(
    google_brain, clean_describer, monkeypatch
):
    """Riding the brain's stack shares the SDK, never the credential."""
    monkeypatch.setenv("HARNESS_DESCRIBER_MODEL", GEMINI)
    monkeypatch.setenv("HARNESS_DESCRIBER_LOCATION", "us")
    assert describer_from_env().fault == "config:missing_credentials_file"


@pytest.mark.parametrize(
    ("env", "fault"),
    [
        ({"HARNESS_DESCRIBER_PROVIDER": "google"}, "config:incomplete_stack"),
        ({"HARNESS_DESCRIBER_SDK": "google-genai"}, "config:incomplete_stack"),
        (
            {"HARNESS_DESCRIBER_PROVIDER": "gemini", "HARNESS_DESCRIBER_SDK": "google-genai"},
            "config:unknown_provider",
        ),
        (
            {"HARNESS_DESCRIBER_PROVIDER": "google", "HARNESS_DESCRIBER_SDK": "google-genai"},
            "config:missing_credentials_file",
        ),
        (
            {
                "HARNESS_DESCRIBER_PROVIDER": "google",
                "HARNESS_DESCRIBER_SDK": "google-genai",
                "HARNESS_DESCRIBER_CREDENTIALS_FILE": "/nonexistent/key.json",
            },
            "config:missing_location",
        ),
        (
            {
                "HARNESS_DESCRIBER_PROVIDER": "google",
                "HARNESS_DESCRIBER_SDK": "google-genai",
                "HARNESS_DESCRIBER_CREDENTIALS_FILE": "/nonexistent/key.json",
                "HARNESS_DESCRIBER_LOCATION": "us",
            },
            "config:missing_credentials_file",
        ),
        (
            {
                "HARNESS_DESCRIBER_PROVIDER": "google",
                "HARNESS_DESCRIBER_SDK": "openai",
            },
            "config:missing_credentials_file",
        ),
    ],
    ids=[
        "provider-without-sdk",
        "sdk-without-provider",
        "unknown-provider",
        "no-key-file",
        "no-location",
        "key-file-absent",
        "google-on-the-wrong-sdk-still-needs-its-key-first",
    ],
)
def test_a_half_configured_describer_is_dead_not_off(
    openrouter_brain, clean_describer, monkeypatch, env, fault
):
    monkeypatch.setenv("HARNESS_DESCRIBER_MODEL", GEMINI)
    for var, value in env.items():
        monkeypatch.setenv(var, value)
    describer = describer_from_env()
    assert describer is not None and describer.provider is None
    assert describer.fault == fault


def test_a_key_file_that_will_not_load_names_its_fault(
    openrouter_brain, clean_describer, monkeypatch, tmp_path
):
    bad = tmp_path / "key.json"
    bad.write_text("{not json")
    monkeypatch.setenv("HARNESS_DESCRIBER_MODEL", GEMINI)
    monkeypatch.setenv("HARNESS_DESCRIBER_PROVIDER", "google")
    monkeypatch.setenv("HARNESS_DESCRIBER_SDK", "google-genai")
    monkeypatch.setenv("HARNESS_DESCRIBER_CREDENTIALS_FILE", str(bad))
    monkeypatch.setenv("HARNESS_DESCRIBER_LOCATION", "us")
    describer = describer_from_env()
    assert describer.fault == "config:invalid_credentials_file"
    assert str(bad) in describer.detail


def test_an_inline_key_for_the_describer_is_refused_and_never_echoed(
    openrouter_brain,
    clean_describer,
    monkeypatch,
    private_key_pem,  # noqa: F811 - fixture
):
    monkeypatch.setenv("HARNESS_DESCRIBER_MODEL", GEMINI)
    monkeypatch.setenv("HARNESS_DESCRIBER_PROVIDER", "google")
    monkeypatch.setenv("HARNESS_DESCRIBER_SDK", "google-genai")
    monkeypatch.setenv("HARNESS_DESCRIBER_CREDENTIALS_FILE", json.dumps(key_info(private_key_pem)))
    monkeypatch.setenv("HARNESS_DESCRIBER_LOCATION", "us")
    describer = describer_from_env()
    assert describer.fault == "config:inline_credentials"
    assert "PRIVATE KEY" not in (describer.detail or "")


def test_an_own_stack_naming_the_brains_provider_keeps_the_brains_endpoint(
    openrouter_brain, clean_describer, monkeypatch
):
    """An OpenRouter regional host stays regional when the describer names OpenRouter itself."""
    monkeypatch.setenv("HARNESS_DESCRIBER_MODEL", "google/gemini-3.8-flash")
    monkeypatch.setenv("HARNESS_DESCRIBER_PROVIDER", "openrouter")
    monkeypatch.setenv("HARNESS_DESCRIBER_SDK", "openrouter")
    monkeypatch.setenv("HARNESS_DESCRIBER_API_KEY", "sk-or-v1-describer0123456789abcdef")
    monkeypatch.setenv("HARNESS_DESCRIBER_PROVIDERS", "google-vertex")
    describer = describer_from_env()
    assert describer.provider.base_url == "https://us.openrouter.ai/api/v1"
    describer.provider.close()


def test_an_own_stack_takes_its_own_surface(google_brain, clean_describer, monkeypatch):
    """OpenRouter over the openai SDK is chat-only, so that stack needs its surface named."""
    monkeypatch.setenv("HARNESS_DESCRIBER_MODEL", "google/gemini-3.8-flash")
    monkeypatch.setenv("HARNESS_DESCRIBER_PROVIDER", "openrouter")
    monkeypatch.setenv("HARNESS_DESCRIBER_SDK", "openai")
    monkeypatch.setenv("HARNESS_DESCRIBER_API_KEY", "sk-or-v1-describer0123456789abcdef")
    monkeypatch.setenv("HARNESS_DESCRIBER_PROVIDERS", "google-vertex")
    assert describer_from_env().fault == "config:no_provider"
    monkeypatch.setenv("HARNESS_DESCRIBER_SDK_SURFACE", "chat")
    describer = describer_from_env()
    assert describer.fault is None and describer.provider.surface == "chat"


def test_a_surface_without_its_stack_is_an_incomplete_stack(
    openrouter_brain, clean_describer, monkeypatch
):
    monkeypatch.setenv("HARNESS_DESCRIBER_MODEL", GEMINI)
    monkeypatch.setenv("HARNESS_DESCRIBER_SDK_SURFACE", "chat")
    assert describer_from_env().fault == "config:incomplete_stack"


def test_the_describer_needs_only_what_its_provider_takes(
    openrouter_brain, clean_describer, monkeypatch
):
    assert describer_required_env() == frozenset()
    monkeypatch.setenv("HARNESS_DESCRIBER_MODEL", GEMINI)
    assert describer_required_env() == {"HARNESS_DESCRIBER_API_KEY"}
    monkeypatch.setenv("HARNESS_DESCRIBER_PROVIDER", "google")
    monkeypatch.setenv("HARNESS_DESCRIBER_SDK", "google-genai")
    assert describer_required_env() == {
        "HARNESS_DESCRIBER_CREDENTIALS_FILE",
        "HARNESS_DESCRIBER_LOCATION",
    }


# === --resolved-config ========================================================================


def test_resolved_config_reports_a_google_brain(
    wake_env,  # noqa: F811
    google_brain,
    clean_describer,
    monkeypatch,
    capsys,
    private_key_pem,  # noqa: F811
):
    monkeypatch.setenv("AI_PROJECT", "nova-project")
    assert main(["--resolved-config"]) == 0
    out = capsys.readouterr().out
    report = json.loads(out)
    assert (report["ai_provider"], report["ai_sdk"], report["ai_sdk_surface"]) == (
        "google",
        "google-genai",
        "native",
    )
    assert report["ai_location"] == "us" and report["ai_project"] == "nova-project"
    assert report["ai_credentials_file"] == str(google_brain)
    assert report["ai_sdk_version"]  # the extra is installed in the dev env
    # The path, never the key inside it.
    assert "PRIVATE KEY" not in out and private_key_pem[40:80] not in out


def test_resolved_config_reports_a_describer_on_its_own_stack(
    wake_env,  # noqa: F811
    clean_describer,
    monkeypatch,
    capsys,
):
    monkeypatch.setenv("HARNESS_DESCRIBER_MODEL", GEMINI)
    monkeypatch.setenv("HARNESS_DESCRIBER_PROVIDER", "google")
    monkeypatch.setenv("HARNESS_DESCRIBER_SDK", "google-genai")
    monkeypatch.setenv("HARNESS_DESCRIBER_CREDENTIALS_FILE", "secrets/vertex-eyes.json")
    assert main(["--resolved-config"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert (report["describer_provider"], report["describer_sdk"]) == ("google", "google-genai")
    assert report["describer_credentials_file"] == "secrets/vertex-eyes.json"
    assert report["describer_location"] is None and report["describer_project"] is None
    assert report["describer_sdk_surface"] is None
    # Every `false` is a capability that cannot do its job: this describer has no location, and its
    # key file is set but does not exist (issue #661), so neither variable reads `true`.
    assert report["describer_credentials_file_state"] == "missing"
    assert report["tool_env"]["HARNESS_DESCRIBER_CREDENTIALS_FILE"] is False
    assert report["tool_env"]["HARNESS_DESCRIBER_LOCATION"] is False
    assert "HARNESS_DESCRIBER_API_KEY" not in report["tool_env"]


# --- the describer's SDK version and the key files' state (issue #661) ------------------------


@pytest.fixture
def vertex_describer(monkeypatch, clean_describer, key_file):  # noqa: F811 - fixture
    monkeypatch.setenv("HARNESS_DESCRIBER_MODEL", GEMINI)
    monkeypatch.setenv("HARNESS_DESCRIBER_PROVIDER", "google")
    monkeypatch.setenv("HARNESS_DESCRIBER_SDK", "google-genai")
    monkeypatch.setenv("HARNESS_DESCRIBER_CREDENTIALS_FILE", str(key_file))
    monkeypatch.setenv("HARNESS_DESCRIBER_LOCATION", "us")
    return key_file


def _report(capsys) -> tuple[dict, str]:
    assert main(["--resolved-config"]) == 0
    out = capsys.readouterr().out
    return json.loads(out), out


def test_the_describer_sdk_version_is_the_installed_one(wake_env, vertex_describer, capsys):  # noqa: F811
    report, _ = _report(capsys)
    assert report["describer_sdk_version"] == metadata.version("google-genai")
    assert report["describer_credentials_file_state"] == "ok"
    assert report["tool_env"]["HARNESS_DESCRIBER_CREDENTIALS_FILE"] is True
    assert report["tool_env"]["HARNESS_DESCRIBER_LOCATION"] is True


def test_a_describer_sdk_that_is_not_installed_reads_null(
    wake_env,  # noqa: F811
    vertex_describer,
    monkeypatch,
    capsys,
):
    """The 2026-10-07 incident: the describer switched to an SDK the box did not have."""
    real = metadata.version

    def version(dist):
        if dist == "google-genai":
            raise metadata.PackageNotFoundError(dist)
        return real(dist)

    monkeypatch.setattr("basecradle_harness._wake.metadata.version", version)
    report, _ = _report(capsys)
    assert report["describer_sdk"] == "google-genai"
    assert "describer_sdk_version" in report and report["describer_sdk_version"] is None


def test_no_describer_sdk_set_reads_null(wake_env, clean_describer, capsys):  # noqa: F811
    report, _ = _report(capsys)
    assert report["describer_sdk"] is None
    assert "describer_sdk_version" in report and report["describer_sdk_version"] is None
    assert report["describer_credentials_file_state"] is None
    assert report["ai_credentials_file_state"] is None


def _write(path, contents: bytes):
    path.write_bytes(contents)
    return path


@pytest.mark.parametrize(
    ("state", "make"),
    [
        ("ok", lambda tmp, key: key),
        ("missing", lambda tmp, key: tmp / "absent.json"),
        ("invalid", lambda tmp, key: _write(tmp / "bad.json", b"not json {")),
    ],
)
def test_each_key_file_is_reported_by_its_state(
    wake_env,  # noqa: F811
    vertex_describer,
    monkeypatch,
    capsys,
    tmp_path,
    state,
    make,
):
    """Both key files, each judged on its own, and `tool_env` true only for a file that loads."""
    path = make(tmp_path, vertex_describer)
    monkeypatch.setenv("HARNESS_DESCRIBER_CREDENTIALS_FILE", str(path))
    monkeypatch.setenv("AI_CREDENTIALS_FILE", str(path))
    report, _ = _report(capsys)
    assert report["describer_credentials_file_state"] == state
    assert report["ai_credentials_file_state"] == state
    assert report["tool_env"]["HARNESS_DESCRIBER_CREDENTIALS_FILE"] is (state == "ok")


def test_a_blank_key_file_variable_reads_false_in_tool_env(
    wake_env,  # noqa: F811
    vertex_describer,
    monkeypatch,
    capsys,
):
    """Set but blank is no key file at all: the describer reads it as missing, and so does the map."""
    monkeypatch.setenv("HARNESS_DESCRIBER_CREDENTIALS_FILE", "   ")
    report, _ = _report(capsys)
    assert report["describer_credentials_file_state"] is None
    assert report["tool_env"]["HARNESS_DESCRIBER_CREDENTIALS_FILE"] is False


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads any file")
def test_an_unreadable_key_file_is_reported_unreadable(
    wake_env,  # noqa: F811
    vertex_describer,
    capsys,
):
    vertex_describer.chmod(0)
    try:
        report, _ = _report(capsys)
    finally:
        vertex_describer.chmod(0o600)
    assert report["describer_credentials_file_state"] == "unreadable"
    assert report["tool_env"]["HARNESS_DESCRIBER_CREDENTIALS_FILE"] is False


def test_without_google_auth_a_good_key_file_is_unchecked_never_ok(
    wake_env,  # noqa: F811
    vertex_describer,
    monkeypatch,
    capsys,
):
    monkeypatch.setitem(sys.modules, "google.oauth2", None)
    monkeypatch.setitem(sys.modules, "google.oauth2.service_account", None)
    report, _ = _report(capsys)
    assert report["describer_credentials_file_state"] == "unchecked"
    assert report["tool_env"]["HARNESS_DESCRIBER_CREDENTIALS_FILE"] is False


SENTINEL = "nova-sentinel-7f3a9c"


def _sentinel_files(tmp_path, pem) -> dict[str, bytes]:
    """A key file in every state, each carrying `SENTINEL` in every field it has."""
    loaded = key_info(
        pem,
        project_id=f"{SENTINEL}-project",
        private_key_id=f"{SENTINEL}0123456789abcdef",
        client_email=f"{SENTINEL}@nova-project.iam.gserviceaccount.com",
        client_id=f"{SENTINEL}-client",
    )
    broken = {**loaded, "private_key": pem.replace("MII", "XXX")}
    return {
        "ok": json.dumps(loaded).encode(),
        "invalid-key": json.dumps(broken).encode(),
        "invalid-type": json.dumps({**loaded, "type": f"{SENTINEL}-type"}).encode(),
        "invalid-json": f'{{"private_key": "{SENTINEL}", '.encode(),
        "invalid-utf8": b"\xff\xfe" + SENTINEL.encode() + b"\x80",
        "invalid-deep": f"[{json.dumps(SENTINEL)}, ".encode() + b"[" * 100_000,
    }


@pytest.mark.parametrize("unchecked", [False, True], ids=["sdk", "no-sdk"])
def test_nothing_read_from_a_key_file_reaches_the_report(
    wake_env,  # noqa: F811
    vertex_describer,
    monkeypatch,
    capsys,
    tmp_path,
    private_key_pem,  # noqa: F811
    unchecked,
):
    if unchecked:
        monkeypatch.setitem(sys.modules, "google.oauth2", None)
        monkeypatch.setitem(sys.modules, "google.oauth2.service_account", None)
    states = set()
    for name, contents in _sentinel_files(tmp_path, private_key_pem).items():
        path = _write(tmp_path / f"{name}.json", contents)
        monkeypatch.setenv("HARNESS_DESCRIBER_CREDENTIALS_FILE", str(path))
        monkeypatch.setenv("AI_CREDENTIALS_FILE", str(path))
        assert main(["--resolved-config"]) == 0
        captured = capsys.readouterr()
        out = captured.out + captured.err
        states.add(json.loads(captured.out)["describer_credentials_file_state"])
        assert SENTINEL not in out
        assert "PRIVATE KEY" not in out and private_key_pem[40:80] not in out
    # The sentinels were read, not skipped: each file reached the loader and got a real verdict.
    assert states == ({"unchecked", "invalid"} if unchecked else {"ok", "invalid"})


def test_resolved_config_withholds_a_key_pasted_where_its_path_belongs(
    wake_env,  # noqa: F811
    clean_describer,
    monkeypatch,
    capsys,
    private_key_pem,  # noqa: F811
):
    inline = json.dumps(key_info(private_key_pem))
    monkeypatch.setenv("AI_CREDENTIALS_FILE", inline)
    monkeypatch.setenv("HARNESS_DESCRIBER_CREDENTIALS_FILE", inline)
    assert main(["--resolved-config"]) == 0
    out = capsys.readouterr().out
    report = json.loads(out)
    assert report["ai_credentials_file"] == "[withheld: not a path]"
    assert report["describer_credentials_file"] == "[withheld: not a path]"
    assert "PRIVATE KEY" not in out and private_key_pem[40:80] not in out


def test_resolved_config_reads_null_for_an_unconfigured_vertex(
    wake_env,  # noqa: F811
    clean_describer,
    monkeypatch,
    capsys,
):
    for var in ("AI_LOCATION", "AI_PROJECT", "AI_CREDENTIALS_FILE"):
        monkeypatch.delenv(var, raising=False)
    assert main(["--resolved-config"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ai_location"] is None and report["ai_credentials_file"] is None
    assert report["describer_provider"] is None and report["describer_sdk"] is None
