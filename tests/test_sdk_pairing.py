"""A known vendor on an SDK that cannot reach it is refused at config time, not at the build.

Issue #846. A native SDK reaches exactly one vendor, and the ``openai`` SDK reaches the four known
vendors it is wired for (and any generic one at its ``AI_BASE_URL``). Only the provider build used
to refuse a known vendor on an SDK that cannot reach it, so ``--resolved-config`` read green and
``basecradle-harness-resolve`` resolved a tool set over a configuration every wake refused — the
case issue #670 closed for generic vendors. These tests pin every pairing, through every reader of
the SDK axis: the config triple, ``--resolved-config``, the off-box resolver, the describer's own
stack, and the build.

The table of which SDK reaches which vendor is written out here by hand, never read from the
module: a test that took its expectations from the table it checks would agree with any table.
"""

from __future__ import annotations

import itertools

import pytest

from basecradle_harness._basecradle import (
    KNOWN_PROVIDERS,
    _config_from_env,
    _provider_from_config,
    sdk_pairing_problem,
)
from basecradle_harness._describer import describer_from_env
from basecradle_harness._resolve import resolve_stems
from basecradle_harness._wake import resolved_config

#: Which known vendors each shipped SDK reaches — the fact under test, stated independently.
REACHES = {
    "openai": {"openai", "xai", "openrouter", "refuseless"},
    "xai-sdk": {"xai"},
    "openrouter": {"openrouter"},
    "google-genai": {"google"},
    "anthropic": {"anthropic"},
}

PAIRS = list(itertools.product(KNOWN_PROVIDERS, REACHES))
REACHABLE = [(provider, sdk) for provider, sdk in PAIRS if provider in REACHES[sdk]]
UNREACHABLE = [(provider, sdk) for provider, sdk in PAIRS if provider not in REACHES[sdk]]

MODEL = "nova-model-1"
FAKE_KEY = "sk-test-0123456789abcdef0123456789abcdef"


def _ids(pairs):
    return [f"{provider}-on-{sdk}" for provider, sdk in pairs]


@pytest.fixture
def stack(monkeypatch):
    """A brain environment with nothing but the stack under test: no surface, host or describer."""
    for var in (
        "AI_SDK_SURFACE",
        "AI_BASE_URL",
        "HARNESS_DESCRIBER_MODEL",
        "HARNESS_DESCRIBER_API_KEY",
        "HARNESS_DESCRIBER_PROVIDERS",
        "HARNESS_DESCRIBER_PROVIDER",
        "HARNESS_DESCRIBER_SDK",
        "HARNESS_DESCRIBER_SDK_SURFACE",
        "HARNESS_DESCRIBER_CREDENTIALS_FILE",
        "HARNESS_DESCRIBER_LOCATION",
        "HARNESS_DESCRIBER_PROJECT",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AI_MODEL", MODEL)
    monkeypatch.setenv("AI_API_KEY", FAKE_KEY)

    def set_stack(provider: str, sdk: str) -> None:
        monkeypatch.setenv("AI_PROVIDER", provider)
        monkeypatch.setenv("AI_SDK", sdk)

    return set_stack


def test_the_matrix_covers_every_known_vendor_and_every_shipped_sdk():
    # A sixth SDK or a seventh vendor must land in `REACHES` with a decided answer, or this file
    # would silently stop covering it.
    from basecradle_harness._basecradle import _NATIVE_SDK_VENDORS, _SDK_SURFACES

    assert set(REACHES) == set(_SDK_SURFACES)
    # The build's openai-SDK tail assumes every other shipped SDK is a row of the pairing table: an
    # SDK added to `_SDK_SURFACES` alone would fall through to it and build an `OpenAIProvider`.
    assert set(_NATIVE_SDK_VENDORS) == set(_SDK_SURFACES) - {"openai"}
    assert all(provider in KNOWN_PROVIDERS for vendors in REACHES.values() for provider in vendors)
    assert {provider for vendors in REACHES.values() for provider in vendors} == set(
        KNOWN_PROVIDERS
    )


@pytest.mark.parametrize(("provider", "sdk"), REACHABLE, ids=_ids(REACHABLE))
def test_a_pairing_that_reaches_its_vendor_passes_every_config_time_reader(stack, provider, sdk):
    stack(provider, sdk)
    assert sdk_pairing_problem(provider, sdk) is None
    assert _config_from_env()[:2] == (provider, sdk)
    assert resolve_stems(provider=provider, sdk=sdk)["ai_provider"] == provider


@pytest.mark.parametrize(("provider", "sdk"), UNREACHABLE, ids=_ids(UNREACHABLE))
def test_a_pairing_that_cannot_reach_its_vendor_is_refused_by_every_reader(stack, provider, sdk):
    stack(provider, sdk)
    problem = sdk_pairing_problem(provider, sdk)
    assert problem is not None
    # The message names the fix: an SDK that does reach this vendor.
    fix = next(name for name, vendors in REACHES.items() if provider in vendors)
    assert f"AI_SDK={fix}" in problem

    for read in (
        _config_from_env,
        resolved_config,
        lambda: resolve_stems(provider=provider, sdk=sdk),
        lambda: _provider_from_config(provider, sdk, ""),
    ):
        with pytest.raises(ValueError) as refused:
            read()
        # One rule, spelled once: every reader refuses in the same words.
        assert str(refused.value) == problem


@pytest.mark.parametrize(
    ("provider", "sdk"),
    [("openai", "anthropic"), ("refuseless", "xai-sdk")],
    ids=["openai-on-anthropic", "refuseless-on-xai-sdk"],
)
def test_the_issues_two_examples_no_longer_read_green(stack, provider, sdk):
    stack(provider, sdk)
    with pytest.raises(ValueError, match=rf"requires AI_PROVIDER=\w+ \(got '{provider}'\)"):
        resolved_config()


def test_a_native_sdk_names_the_vendor_it_reaches_and_how_the_configured_one_is_reached():
    assert sdk_pairing_problem("refuseless", "xai-sdk") == (
        "AI_SDK=xai-sdk reaches xAI's native endpoint, so it requires AI_PROVIDER=xai (got "
        "'refuseless'). AI_PROVIDER='refuseless' is reached with AI_SDK=openai."
    )
    # A vendor with two cells names both.
    assert sdk_pairing_problem("xai", "anthropic").endswith(
        "AI_PROVIDER='xai' is reached with AI_SDK=xai-sdk or AI_SDK=openai."
    )


def test_the_openai_sdk_names_the_native_sdk_a_vendor_it_does_not_reach_needs():
    problem = sdk_pairing_problem("anthropic", "openai")
    assert problem.startswith("AI_PROVIDER='anthropic' is not reached through the openai SDK")
    assert problem.endswith("AI_PROVIDER='anthropic' is reached with AI_SDK=anthropic.")


def test_an_sdk_with_no_adapter_is_refused_at_config_time_for_a_known_vendor(stack):
    stack("openai", "cohere")
    with pytest.raises(ValueError, match="AI_SDK='cohere' has no adapter"):
        _config_from_env()
    with pytest.raises(ValueError, match="AI_SDK='cohere' has no adapter"):
        resolved_config()


@pytest.mark.parametrize(("provider", "sdk"), UNREACHABLE, ids=_ids(UNREACHABLE))
def test_a_describers_own_stack_is_refused_in_its_own_variables_names(
    stack, monkeypatch, provider, sdk
):
    # The brain is a valid stack; only the describer's own is not. Its fault must name the
    # describer's variables, never tell the operator to change the brain's.
    stack("openai", "openai")
    monkeypatch.setenv("HARNESS_DESCRIBER_MODEL", "nova-vision-1")
    monkeypatch.setenv("HARNESS_DESCRIBER_PROVIDER", provider)
    monkeypatch.setenv("HARNESS_DESCRIBER_SDK", sdk)
    describer = describer_from_env()
    assert describer is not None and describer.provider is None
    assert describer.fault == "config:no_provider"
    assert describer.detail == sdk_pairing_problem(
        provider, sdk, provider_var="HARNESS_DESCRIBER_PROVIDER", sdk_var="HARNESS_DESCRIBER_SDK"
    )
    # Independently of the rule's own output: the fix it names is in the describer's variable.
    fix = next(name for name, vendors in REACHES.items() if provider in vendors)
    assert f"HARNESS_DESCRIBER_SDK={fix}" in describer.detail
    assert "AI_SDK=" not in describer.detail
    assert "AI_PROVIDER=" not in describer.detail
