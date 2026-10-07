"""The Google adapter (`GoogleProvider`) — Gemini on Vertex AI over the real ``google-genai`` SDK.

Every test drives the **real SDK** in Vertex mode. The adapter hands the SDK its own ``httpx`` client,
which respx intercepts, so what is asserted is the request the SDK actually serialized — the URL,
the headers, the per-phase timeout, the JSON body — never what the adapter *meant* to send (issue
#433's rule: a field a Python signature accepts is not a field on the wire).

The one call ``google-auth`` makes outside ``httpx`` is the service-account token refresh, over
``requests``. It is stubbed to mint a fake token, and the ``requests`` transport and Application
Default Credentials are both made to fail loudly if anything reaches them — so a test that passed by
quietly borrowing a credential from the machine it runs on cannot exist.

The service-account key is generated per session and fabricated in every field (project
``nova-project``, ``nova@nova-project.iam.gserviceaccount.com``).
"""

from __future__ import annotations

import base64
import datetime
import json
import logging
import os
import re
import sys
from pathlib import Path

import httpx
import pytest
import respx
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from google.auth import exceptions as auth_errors
from google.oauth2 import service_account

from basecradle_harness import (
    GoogleSearchTool,
    ImageContent,
    Message,
    ProviderAPIError,
    ProviderAuthError,
    ProviderBillingError,
    ProviderConnectionError,
    ProviderContextLengthError,
    ProviderError,
    ProviderPayloadTooLargeError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderServerError,
    ProviderTimeoutError,
    ToolSpec,
    VideoContent,
)
from basecradle_harness._caching import AUTOMATIC
from basecradle_harness._google import (
    LOCAL_CALL_ID_PREFIX,
    SKIP_SIGNATURE,
    SYSTEM_LABEL,
    UNCHECKED,
    GoogleConfigError,
    GoogleProvider,
    credentials_file_state,
)
from basecradle_harness._google_rates import Usage, call_cost
from basecradle_harness._observability import truncated
from basecradle_harness._timeouts import CONNECT_TIMEOUT, TIMEOUT_RETRY_SCALE

MODEL = "gemini-3.8-flash"
PROJECT = "nova-project"
LOCATION = "us"
HOST = "https://aiplatform.us.rep.googleapis.com"
GENERATE = (
    f"{HOST}/v1beta1/projects/{PROJECT}/locations/{LOCATION}/publishers/google/models/"
    f"{MODEL}:generateContent"
)
FAKE_TOKEN = "ya29.fake-vertex-token-0123456789"

MEMORY_TOOL = ToolSpec(
    name="memory_search",
    description="Search memory.",
    parameters={"type": "object", "properties": {"query": {"type": "string"}}},
)


# --- the credential --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def private_key_pem() -> str:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()


def key_info(pem: str, **overrides) -> dict:
    """A fabricated service-account key, every field fake."""
    return {
        "type": "service_account",
        "project_id": PROJECT,
        "private_key_id": "0123456789abcdef0123456789abcdef01234567",
        "private_key": pem,
        "client_email": f"nova@{PROJECT}.iam.gserviceaccount.com",
        "client_id": "100000000000000000001",
        "token_uri": "https://oauth2.googleapis.com/token",
        **overrides,
    }


@pytest.fixture
def key_file(tmp_path, private_key_pem) -> Path:
    path = tmp_path / "vertex-key.json"
    path.write_text(json.dumps(key_info(private_key_pem)))
    return path


@pytest.fixture(autouse=True)
def _no_real_google_auth(monkeypatch):
    """Mint a fake token on refresh, and make every road to a real Google credential a failure."""

    def refresh(self, request):
        self.token = FAKE_TOKEN
        self.expiry = datetime.datetime.now(datetime.timezone.utc).replace(
            tzinfo=None
        ) + datetime.timedelta(hours=1)

    def forbidden(*args, **kwargs):
        raise AssertionError("a test reached a real Google credential or endpoint")

    monkeypatch.setattr(service_account.Credentials, "refresh", refresh)
    monkeypatch.setattr("google.auth.default", forbidden)
    monkeypatch.setattr("google.auth.transport.requests.Request.__call__", forbidden)
    for var in ("AI_CREDENTIALS_FILE", "AI_LOCATION", "AI_PROJECT", "AI_BASE_URL"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def router():
    with respx.mock(assert_all_mocked=True, assert_all_called=False) as r:
        yield r


def build(key_file: Path, **overrides) -> GoogleProvider:
    fields = {"credentials_file": str(key_file), "location": LOCATION}
    return GoogleProvider(MODEL, **{**fields, **overrides})


@pytest.fixture
def provider(key_file):
    p = build(key_file)
    yield p
    p.close()


# --- response bodies -------------------------------------------------------------------------


def body(*parts, finish="STOP", usage=None, response_id="resp-0123456789abcdef"):
    """A ``generateContent`` response body as Vertex sends it."""
    return {
        "candidates": [
            {"content": {"role": "model", "parts": list(parts)}, "finishReason": finish}
        ],
        "usageMetadata": usage
        if usage is not None
        else {
            "promptTokenCount": 1200,
            "candidatesTokenCount": 30,
            "thoughtsTokenCount": 70,
            "totalTokenCount": 1300,
            "cachedContentTokenCount": 1000,
            "trafficType": "ON_DEMAND",
        },
        "responseId": response_id,
        "modelVersion": MODEL,
    }


def text(value: str) -> dict:
    return {"text": value}


def call(name: str, args: dict, *, id: str | None = None, signature: bytes | None = None) -> dict:
    part: dict = {"functionCall": {"name": name, "args": args}}
    if id is not None:
        part["functionCall"]["id"] = id
    if signature is not None:
        part["thoughtSignature"] = base64.b64encode(signature).decode()
    return part


def sent(route) -> dict:
    return json.loads(route.calls.last.request.content)


def ok(*parts, **kwargs) -> httpx.Response:
    return httpx.Response(200, json=body(*parts, **kwargs))


# === configuration =============================================================================


def test_location_is_required_and_has_no_default(key_file):
    with pytest.raises(GoogleConfigError) as raised:
        GoogleProvider(MODEL, credentials_file=str(key_file))
    assert raised.value.reason == "missing_location"
    assert "global" in str(raised.value)


def test_a_credential_file_is_required(monkeypatch):
    with pytest.raises(GoogleConfigError) as raised:
        GoogleProvider(MODEL, location=LOCATION)
    assert raised.value.reason == "missing_credentials_file"


def test_a_missing_credential_file_names_the_path(tmp_path):
    missing = tmp_path / "absent.json"
    with pytest.raises(GoogleConfigError) as raised:
        GoogleProvider(MODEL, credentials_file=str(missing), location=LOCATION)
    assert raised.value.reason == "missing_credentials_file"
    assert str(missing) in str(raised.value)


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads any file")
def test_an_unreadable_credential_file_is_its_own_fault(key_file):
    key_file.chmod(0)
    try:
        with pytest.raises(GoogleConfigError) as raised:
            build(key_file)
    finally:
        key_file.chmod(0o600)
    assert raised.value.reason == "unreadable_credentials_file"


def test_a_credential_file_that_is_not_json_is_invalid(tmp_path):
    bad = tmp_path / "key.json"
    bad.write_text("not json {")
    with pytest.raises(GoogleConfigError) as raised:
        build(bad)
    assert raised.value.reason == "invalid_credentials_file"


def test_a_users_own_login_is_refused_never_borrowed(tmp_path):
    """Only a service-account key loads. A ``gcloud`` user credential is someone's personal account."""
    user = tmp_path / "adc.json"
    user.write_text(
        json.dumps(
            {
                "type": "authorized_user",
                "client_id": "x",
                "client_secret": "y",
                "refresh_token": "1//fake",
            }
        )
    )
    with pytest.raises(GoogleConfigError) as raised:
        build(user)
    assert raised.value.reason == "invalid_credentials_file"


def test_a_broken_key_never_puts_the_key_in_the_error(tmp_path, private_key_pem):
    broken = tmp_path / "key.json"
    info = key_info(private_key_pem.replace("MII", "XXX"))
    broken.write_text(json.dumps(info))
    with pytest.raises(GoogleConfigError) as raised:
        build(broken)
    assert raised.value.reason == "invalid_credentials_file"
    # google-auth's `InvalidValue` quotes the key in its repr, so neither link may hold it.
    assert raised.value.__cause__ is None and raised.value.__context__ is None
    message = str(raised.value)
    assert "PRIVATE KEY" not in message and info["private_key"][40:80] not in message


def test_a_key_pasted_where_its_path_belongs_is_refused_and_never_echoed(private_key_pem):
    """Many Google tools take the JSON inline; here it must be a path, and every error names it."""
    inline = json.dumps(key_info(private_key_pem))
    with pytest.raises(GoogleConfigError) as raised:
        GoogleProvider(MODEL, credentials_file=inline, location=LOCATION)
    assert raised.value.reason == "inline_credentials"
    assert "PRIVATE KEY" not in str(raised.value) and PROJECT not in str(raised.value)


def test_a_key_file_that_is_not_text_is_invalid_and_never_quoted(tmp_path):
    """A binary file is read and is not UTF-8: a config fault like any other, never a raw decode error."""
    binary = tmp_path / "key.json"
    binary.write_bytes(b"\xff\xfe nova-secret-bytes \x80")
    with pytest.raises(GoogleConfigError) as raised:
        build(binary)
    assert raised.value.reason == "invalid_credentials_file"
    assert "nova-secret-bytes" not in str(raised.value)
    # Neither link of the chain: a decode error's `.object` is the file's bytes.
    assert raised.value.__cause__ is None and raised.value.__context__ is None


def test_a_key_file_that_is_not_json_keeps_no_copy_of_itself_on_the_error(tmp_path):
    """A JSON error's ``.doc`` is the whole document, so it must not ride the chain either."""
    bad = tmp_path / "key.json"
    bad.write_text('{"private_key": "nova-secret-doc", ')
    with pytest.raises(GoogleConfigError) as raised:
        build(bad)
    assert raised.value.reason == "invalid_credentials_file"
    assert raised.value.__cause__ is None and raised.value.__context__ is None


def test_a_deeply_nested_key_file_is_invalid_not_a_crash(tmp_path):
    """`json.loads` raises `RecursionError`, a ``RuntimeError``, on a deep enough document."""
    deep = tmp_path / "key.json"
    deep.write_text("[" * 100_000)
    with pytest.raises(GoogleConfigError) as raised:
        build(deep)
    assert raised.value.reason == "invalid_credentials_file"
    assert credentials_file_state(str(deep)) == "invalid"


def test_a_key_the_crypto_library_cannot_handle_is_invalid_not_a_crash(
    monkeypatch, key_file, private_key_pem
):
    """`cryptography` raises `UnsupportedAlgorithm`, outside the `ValueError` family, for a key on a
    curve it does not support; whatever the parse raises is the one config fault."""
    from cryptography.exceptions import UnsupportedAlgorithm

    def unsupported(info, **kwargs):
        raise UnsupportedAlgorithm(f"unsupported curve in {info['private_key']}")

    monkeypatch.setattr(service_account.Credentials, "from_service_account_info", unsupported)
    with pytest.raises(GoogleConfigError) as raised:
        build(key_file)
    assert raised.value.reason == "invalid_credentials_file"
    assert "UnsupportedAlgorithm" in str(raised.value)
    assert raised.value.__cause__ is None and raised.value.__context__ is None
    assert credentials_file_state(str(key_file)) == "invalid"


def test_a_home_directory_that_does_not_exist_is_invalid_not_a_crash():
    raw = "~nova-no-such-user-7f3a9c/key.json"
    with pytest.raises(GoogleConfigError) as raised:
        GoogleProvider(MODEL, credentials_file=raw, location=LOCATION)
    assert raised.value.reason == "invalid_credentials_file"
    assert credentials_file_state(raw) == "invalid"


# --- the key file's state, as --resolved-config reports it (issue #661) -------------------------


def test_the_state_of_a_key_that_loads_is_ok(key_file):
    assert credentials_file_state(str(key_file)) == "ok"


@pytest.mark.parametrize("raw", [None, "", "   "])
def test_an_unset_key_file_has_no_state(raw):
    assert credentials_file_state(raw) is None


def test_the_state_of_an_absent_key_file_is_missing(tmp_path):
    assert credentials_file_state(str(tmp_path / "absent.json")) == "missing"


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads any file")
def test_the_state_of_a_key_file_that_cannot_be_read_is_unreadable(key_file):
    key_file.chmod(0)
    try:
        assert credentials_file_state(str(key_file)) == "unreadable"
    finally:
        key_file.chmod(0o600)


def test_a_directory_where_the_key_file_belongs_is_unreadable(tmp_path):
    assert credentials_file_state(str(tmp_path)) == "unreadable"


@pytest.mark.parametrize(
    "contents",
    [
        b"not json {",
        b"\xff\xfe not text \x80",
        json.dumps({"type": "authorized_user", "refresh_token": "1//fake"}).encode(),
        json.dumps(["a", "list"]).encode(),
    ],
    ids=["not-json", "not-utf8", "not-a-service-account", "not-an-object"],
)
def test_the_state_of_a_key_file_that_will_not_load_is_invalid(tmp_path, contents):
    bad = tmp_path / "key.json"
    bad.write_bytes(contents)
    assert credentials_file_state(str(bad)) == "invalid"


def test_the_state_of_a_key_whose_private_key_will_not_parse_is_invalid(tmp_path, private_key_pem):
    broken = tmp_path / "key.json"
    broken.write_text(json.dumps(key_info(private_key_pem.replace("MII", "XXX"))))
    assert credentials_file_state(str(broken)) == "invalid"


def test_the_state_of_a_key_pasted_where_its_path_belongs_is_invalid(private_key_pem):
    assert credentials_file_state(json.dumps(key_info(private_key_pem))) == "invalid"


def test_the_state_is_the_wakes_own_verdict(tmp_path, private_key_pem):
    """One loader, one verdict: every file the adapter refuses has a state that is not ``ok``."""
    cases = {
        "missing": tmp_path / "absent.json",
        "invalid": tmp_path / "bad.json",
    }
    cases["invalid"].write_text("not json {")
    for state, path in cases.items():
        with pytest.raises(GoogleConfigError):
            build(path)
        assert credentials_file_state(str(path)) == state


def test_a_relative_key_file_is_judged_where_the_wake_reads_it(
    monkeypatch, tmp_path, private_key_pem
):
    """A relative path resolves against the config home, exactly as `credentials_path` resolves it."""
    home = tmp_path / "config-home"
    (home / "secrets").mkdir(parents=True)
    (home / "secrets" / "vertex.json").write_text(json.dumps(key_info(private_key_pem)))
    monkeypatch.setenv("BASECRADLE_CONFIG_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    assert credentials_file_state("secrets/vertex.json") == "ok"


def test_without_google_auth_the_file_is_still_judged_and_the_key_is_unchecked(
    monkeypatch, tmp_path, key_file
):
    """The SDK missing (the 2026-10-07 incident) leaves every file check but the key parse runnable.

    A file that fails those checks reports its real fault; one that passes them is ``unchecked``,
    never ``ok``, because nobody parsed the key.
    """
    monkeypatch.setitem(sys.modules, "google.oauth2", None)
    monkeypatch.setitem(sys.modules, "google.oauth2.service_account", None)
    assert credentials_file_state(str(key_file)) == UNCHECKED
    assert credentials_file_state(str(tmp_path / "absent.json")) == "missing"
    bad = tmp_path / "bad.json"
    bad.write_text("not json {")
    assert credentials_file_state(str(bad)) == "invalid"


def test_the_project_defaults_to_the_one_the_key_names(provider):
    assert provider.project == PROJECT


def test_an_explicit_project_wins(router, key_file):
    route = router.post(GENERATE.replace(f"projects/{PROJECT}", "projects/nova-other")).mock(
        return_value=ok(text("hi"))
    )
    with build(key_file, project="nova-other") as p:
        p.chat([Message.user("hello")])
    assert route.called


def test_no_project_anywhere_is_a_fault(tmp_path, private_key_pem):
    bare = tmp_path / "key.json"
    info = key_info(private_key_pem)
    del info["project_id"]
    bare.write_text(json.dumps(info))
    with pytest.raises(GoogleConfigError) as raised:
        build(bare)
    assert raised.value.reason == "missing_project"


def test_the_brain_reads_its_vertex_configuration_from_the_environment(monkeypatch, key_file):
    monkeypatch.setenv("AI_CREDENTIALS_FILE", str(key_file))
    monkeypatch.setenv("AI_LOCATION", "us")
    monkeypatch.setenv("AI_PROJECT", "nova-env")
    with GoogleProvider(MODEL) as p:
        assert (p.location, p.project) == ("us", "nova-env")


def test_a_key_file_given_explicitly_takes_nothing_from_the_brains_environment(
    monkeypatch, key_file
):
    """The describer passes its own key file; the brain's location and project must not ride in."""
    monkeypatch.setenv("AI_LOCATION", "eu")
    monkeypatch.setenv("AI_PROJECT", "nova-brain")
    with pytest.raises(GoogleConfigError) as raised:
        GoogleProvider(MODEL, credentials_file=str(key_file))
    assert raised.value.reason == "missing_location"
    with GoogleProvider(MODEL, credentials_file=str(key_file), location="us") as p:
        assert (p.location, p.project) == ("us", PROJECT)


def test_a_relative_key_path_is_read_from_the_config_home(monkeypatch, private_key_pem):
    home = Path(os.environ["BASECRADLE_CONFIG_HOME"])
    (home / "secrets").mkdir(parents=True, exist_ok=True)
    (home / "secrets" / "vertex.json").write_text(json.dumps(key_info(private_key_pem)))
    with GoogleProvider(MODEL, credentials_file="secrets/vertex.json", location="us") as p:
        assert p.project == PROJECT


def test_a_location_is_read_case_blind_so_routing_and_pricing_agree(router, key_file):
    """The SDK compares the location case-sensitively; ``US`` must not become ``US-aiplatform``."""
    route = router.post(GENERATE).mock(return_value=ok(text("hi")))
    with build(key_file, location=" US ") as p:
        p.chat([Message.user("hello")])
        assert p.location == "us"
    assert route.called


def test_a_gemini_api_only_setting_fails_at_startup(key_file):
    """The typed config accepts it; the SDK then refuses to send it to Vertex on every call."""
    with pytest.raises(GoogleConfigError) as raised:
        build(key_file, enable_enhanced_civic_answers=True)
    assert raised.value.reason == "invalid_model_params"


def test_a_nested_setting_vertex_will_not_take_fails_at_startup(key_file):
    """Asked of the SDK's own Vertex serializer, so a field no hand-kept list names is caught too."""
    with pytest.raises(GoogleConfigError) as raised:
        build(key_file, tool_config={"include_server_side_tool_invocations": True})
    assert raised.value.reason == "invalid_model_params"


def test_the_sdk_refusing_a_request_is_a_provider_error_not_a_raw_valueerror():
    class Refusing:
        class models:
            @staticmethod
            def generate_content(**kwargs):
                raise ValueError("only supported in Gemini Developer API mode")

    with pytest.raises(ProviderError, match="refused the request") as raised:
        GoogleProvider(MODEL, client=Refusing(), location=LOCATION).chat([Message.user("hi")])
    assert not isinstance(raised.value, ProviderAPIError)


def test_an_unknown_model_param_fails_at_startup_not_mid_wake(key_file):
    with pytest.raises(GoogleConfigError) as raised:
        build(key_file, temperture=0.2)
    assert raised.value.reason == "invalid_model_params"
    assert "model_params.json" in str(raised.value)


def test_application_default_credentials_are_never_consulted(router, key_file):
    """`google.auth.default` raises in this module — a call still works, so it was never asked."""
    router.post(GENERATE).mock(return_value=ok(text("fine")))
    with build(key_file) as p:
        assert p.chat([Message.user("hi")]).content == "fine"


# === the wire ==================================================================================


@pytest.mark.parametrize(
    ("location", "host"),
    [
        ("us", "https://aiplatform.us.rep.googleapis.com"),
        ("eu", "https://aiplatform.eu.rep.googleapis.com"),
        ("global", "https://aiplatform.googleapis.com"),
        ("us-central1", "https://us-central1-aiplatform.googleapis.com"),
    ],
)
def test_the_location_decides_the_endpoint(router, key_file, location, host):
    """``us`` is Google's multi-region endpoint — the residency the fleet chose this provider for."""
    path = (
        f"/v1beta1/projects/{PROJECT}/locations/{location}/publishers/google/models/"
        f"{MODEL}:generateContent"
    )
    route = router.post(host + path).mock(return_value=ok(text("hi")))
    with build(key_file, location=location) as p:
        p.chat([Message.user("hello")])
    assert route.called


def test_the_call_carries_the_service_accounts_token(router, provider):
    route = router.post(GENERATE).mock(return_value=ok(text("hi")))
    provider.chat([Message.user("hello")])
    assert route.calls.last.request.headers["authorization"] == f"Bearer {FAKE_TOKEN}"


def test_an_endpoint_override_is_honoured(router, key_file):
    route = router.post(url__regex=r"https://vertex\.proxy\.test/.*:generateContent").mock(
        return_value=ok(text("hi"))
    )
    with build(key_file, base_url="https://vertex.proxy.test/") as p:
        p.chat([Message.user("hello")])
    assert route.called


def test_each_phase_gets_its_own_budget_on_the_wire(router, provider):
    """The fixed connect and the fitted generation, read off the request itself (issue #589)."""
    route = router.post(GENERATE).mock(return_value=ok(text("hi")))
    provider.chat([Message.user("hello")])
    request = route.calls.last.request
    timeout = request.extensions["timeout"]
    assert timeout["connect"] == CONNECT_TIMEOUT and timeout["pool"] == CONNECT_TIMEOUT
    assert timeout["read"] == timeout["write"] == provider.last_timeout
    assert provider.last_timeout >= 60
    # The SDK tells Vertex the same deadline, so the server stops when the client does.
    assert request.headers["x-server-timeout"] == str(int(provider.last_timeout))


def test_the_retry_scale_reaches_the_wire(router, provider):
    route = router.post(GENERATE).mock(return_value=ok(text("hi")))
    provider.chat([Message.user("hello")])
    first = route.calls.last.request.extensions["timeout"]["read"]
    provider.bind_timeout_scale(TIMEOUT_RETRY_SCALE)
    provider.chat([Message.user("hello")])
    assert route.calls.last.request.extensions["timeout"]["read"] == first * TIMEOUT_RETRY_SCALE


def test_the_sdk_never_retries_on_its_own(router, provider):
    """One request per call: the engine's bounded retry is the only retry policy (`_retry`)."""
    route = router.post(GENERATE).mock(return_value=httpx.Response(503, json={"error": {}}))
    with pytest.raises(ProviderServerError):
        provider.chat([Message.user("hello")])
    assert route.call_count == 1


def test_only_the_leading_system_turns_become_the_system_instruction(router, provider):
    """The brief stays at the tail, where it is volatile — the cache invariant (Context Discipline)."""
    route = router.post(GENERATE).mock(return_value=ok(text("hi")))
    provider.chat(
        [
            Message.system("charter"),
            Message.system("earlier summary"),
            Message.user("hello"),
            Message.assistant("hi there"),
            Message.system("the per-wake brief"),
            Message.user("what now?"),
        ]
    )
    request = sent(route)
    assert request["systemInstruction"]["parts"] == [{"text": "charter\n\nearlier summary"}]
    contents = request["contents"]
    assert [c["role"] for c in contents] == ["user", "model", "user", "user"]
    assert contents[2]["parts"] == [{"text": SYSTEM_LABEL + "the per-wake brief"}]
    assert contents[3]["parts"] == [{"text": "what now?"}]


def test_tools_are_declared_with_their_schema_as_written(router, provider):
    route = router.post(GENERATE).mock(return_value=ok(text("hi")))
    provider.chat([Message.user("hello")], [MEMORY_TOOL])
    declared = sent(route)["tools"][0]["functionDeclarations"][0]
    assert declared["name"] == "memory_search"
    assert declared["description"] == "Search memory."
    assert declared["parameters_json_schema"] == MEMORY_TOOL.parameters


def test_no_tools_sends_no_tools_key(router, provider):
    route = router.post(GENERATE).mock(return_value=ok(text("hi")))
    provider.chat([Message.user("hello")])
    assert "tools" not in sent(route)


def test_a_steps_results_travel_together_and_answer_by_name(router, provider):
    """Parallel results become one user content, in call order — interleaving them is a 400."""
    route = router.post(GENERATE).mock(return_value=ok(text("done")))
    provider.chat(
        [
            Message.user("search twice"),
            Message.assistant(
                tool_calls=[
                    _call("call-a", "memory_search", {"query": "a"}),
                    _call("call-b", "web_fetch", {"url": "https://example.com"}),
                ]
            ),
            Message.tool("call-a", "found a"),
            Message.tool("call-b", "fetched b"),
        ]
    )
    model_turn, results = sent(route)["contents"][1:]
    assert [p["functionCall"]["name"] for p in model_turn["parts"]] == [
        "memory_search",
        "web_fetch",
    ]
    assert results["role"] == "user"
    assert results["parts"] == [
        {
            "functionResponse": {
                "id": "call-a",
                "name": "memory_search",
                "response": {"result": "found a"},
            }
        },
        {
            "functionResponse": {
                "id": "call-b",
                "name": "web_fetch",
                "response": {"result": "fetched b"},
            }
        },
    ]


def test_a_call_id_the_adapter_made_up_is_never_sent_back(router, provider):
    route = router.post(GENERATE).mock(return_value=ok(text("done")))
    local = f"{LOCAL_CALL_ID_PREFIX}0123"
    provider.chat(
        [
            Message.user("go"),
            Message.assistant(tool_calls=[_call(local, "memory_search", {"query": "a"})]),
            Message.tool(local, "found"),
        ]
    )
    model_turn, results = sent(route)["contents"][1:]
    assert "id" not in model_turn["parts"][0]["functionCall"]
    assert "id" not in results["parts"][0]["functionResponse"]


def test_images_and_clips_travel_inline(router, provider):
    route = router.post(GENERATE).mock(return_value=ok(text("I see it")))
    png = base64.b64encode(b"\x89PNG fake").decode()
    mp4 = base64.b64encode(b"fake mp4 bytes").decode()
    provider.chat(
        [
            Message(
                role="user",
                content="look",
                images=[ImageContent(url=f"data:image/png;base64,{png}", alt="a.png")],
                videos=[VideoContent(url=f"data:video/mp4;base64,{mp4}", alt="b.mp4")],
            )
        ]
    )
    parts = sent(route)["contents"][0]["parts"]
    assert parts[0] == {"text": "look"}
    assert parts[1]["inlineData"] == {"data": png, "mime_type": "image/png"}
    assert parts[2]["inlineData"] == {"data": mp4, "mime_type": "video/mp4"}


def test_model_params_reach_the_generation_config(router, key_file):
    route = router.post(GENERATE).mock(return_value=ok(text("hi")))
    with build(
        key_file,
        temperature=0.2,
        max_output_tokens=2048,
        thinking_config={"thinking_level": "HIGH"},
        labels={"agent": "nova"},
    ) as p:
        p.chat([Message.user("hello")])
    request = sent(route)
    config = request["generationConfig"]
    assert config["temperature"] == 0.2 and config["maxOutputTokens"] == 2048
    # Read case-blind: inside `thinkingConfig` the SDK itself writes the field's proto name
    # (`thinking_level`), which Vertex's JSON parser accepts alongside the camelCase form.
    thinking = {
        key.replace("_", "").lower(): value for key, value in config["thinkingConfig"].items()
    }
    assert thinking == {"thinkinglevel": "HIGH"}
    assert request["labels"] == {"agent": "nova"}


def test_extra_body_is_merged_into_the_request(router, key_file):
    route = router.post(GENERATE).mock(return_value=ok(text("hi")))
    with build(key_file, extra_body={"futureField": {"on": True}}) as p:
        p.chat([Message.user("hello")])
        assert p.tuning["extra_body"] == {"futureField": {"on": True}}
    assert sent(route)["futureField"] == {"on": True}


def test_tuning_is_a_deep_copy(key_file):
    with build(key_file, thinking_config={"thinking_level": "HIGH"}) as p:
        p.tuning["thinking_config"]["thinking_level"] = "LOW"
        assert p.tuning == {"thinking_config": {"thinking_level": "HIGH"}}


# === thought signatures ========================================================================


def test_a_signature_goes_back_on_the_call_it_came_with(router, provider):
    route = router.post(GENERATE).mock(
        side_effect=[
            ok(call("memory_search", {"query": "x"}, id="fc-1", signature=b"sig-A")),
            ok(text("answer")),
        ]
    )
    history = [Message.user("find x")]
    reply = provider.chat(history, [MEMORY_TOOL])
    history += [reply, Message.tool(reply.tool_calls[0].id, "x is 1")]
    # The engine appends a step note after a step's results; to Google that is a new user turn.
    history.append(Message.system("step 2 of 24"))
    provider.chat(history, [MEMORY_TOOL])
    model_turn = sent(route)["contents"][1]
    assert model_turn["parts"][0]["thoughtSignature"] == base64.b64encode(b"sig-A").decode()


def test_a_parallel_batch_signs_its_first_call_only(router, provider):
    route = router.post(GENERATE).mock(
        side_effect=[
            ok(
                call("memory_search", {"query": "a"}, id="fc-1", signature=b"sig-P"),
                call("memory_search", {"query": "b"}, id="fc-2"),
            ),
            ok(text("answer")),
        ]
    )
    history = [Message.user("two searches")]
    reply = provider.chat(history, [MEMORY_TOOL])
    history += [reply, Message.tool("fc-1", "a"), Message.tool("fc-2", "b")]
    provider.chat(history, [MEMORY_TOOL])
    first, second = sent(route)["contents"][1]["parts"]
    assert first["thoughtSignature"] == base64.b64encode(b"sig-P").decode()
    assert "thoughtSignature" not in second


def test_a_current_turn_call_with_no_signature_held_carries_the_documented_bypass(router, provider):
    """A turn resumed by a later wake: its call's signature died with the wake that held it."""
    route = router.post(GENERATE).mock(return_value=ok(text("answer")))
    provider.chat(
        [
            Message.user("find x"),
            Message.assistant(tool_calls=[_call("fc-old", "memory_search", {"query": "x"})]),
            Message.tool("fc-old", "x is 1"),
        ],
        [MEMORY_TOOL],
    )
    model_turn = sent(route)["contents"][1]
    assert model_turn["parts"][0]["thoughtSignature"] == base64.b64encode(SKIP_SIGNATURE).decode()


def test_a_reused_vendor_id_never_hands_one_step_anothers_signature(router, provider):
    """Ids are unique only within a response; a model numbering per response reuses ``call_0``."""
    route = router.post(GENERATE).mock(
        side_effect=[
            ok(call("memory_search", {"query": "a"}, id="call_0", signature=b"sig-1")),
            ok(call("memory_search", {"query": "b"}, id="call_0", signature=b"sig-2")),
            ok(text("done")),
        ]
    )
    history = [Message.user("two steps")]
    for result in ("a found", "b found"):
        reply = provider.chat(history, [MEMORY_TOOL])
        history += [reply, Message.tool("call_0", result), Message.system("step note")]
    # Rebuilt from its serialized form, as a transcript read back from disk is: the call objects
    # the adapter returned are gone, so the signatures are found by the calls' content alone.
    provider.chat([Message.from_dict(m.to_dict()) for m in history], [MEMORY_TOOL])
    contents = sent(route)["contents"]
    signed = [c["parts"][0].get("thoughtSignature") for c in contents if c["role"] == "model"]
    assert signed == [base64.b64encode(b"sig-1").decode(), base64.b64encode(b"sig-2").decode()]


def test_a_tool_that_mutates_its_arguments_keeps_its_calls_signature(router, provider):
    route = router.post(GENERATE).mock(
        side_effect=[
            ok(call("memory_search", {"query": "x", "tags": ["b", "a"]}, signature=b"sig-M")),
            ok(text("done")),
        ]
    )
    history = [Message.user("search")]
    reply = provider.chat(history, [MEMORY_TOOL])
    reply.tool_calls[0].arguments["tags"].sort()  # a tool sorting its own nested argument in place
    history += [reply, Message.tool(reply.tool_calls[0].id, "found"), Message.system("step note")]
    provider.chat(history, [MEMORY_TOOL])
    model_turn = sent(route)["contents"][1]
    assert model_turn["parts"][0]["thoughtSignature"] == base64.b64encode(b"sig-M").decode()


def test_an_older_turns_call_with_no_signature_goes_without(router, provider):
    """History is not validated, and Google says not to dress it in fake signatures."""
    route = router.post(GENERATE).mock(return_value=ok(text("answer")))
    provider.chat(
        [
            Message.user("find x"),
            Message.assistant(tool_calls=[_call("fc-old", "memory_search", {"query": "x"})]),
            Message.tool("fc-old", "x is 1"),
            Message.assistant("x is 1."),
            Message.user("thanks — and y?"),
        ],
        [MEMORY_TOOL],
    )
    model_turn = sent(route)["contents"][1]
    assert "thoughtSignature" not in model_turn["parts"][0]


# === reading the answer ========================================================================


def test_text_parts_join_and_thought_summaries_are_left_out(router, provider):
    router.post(GENERATE).mock(
        return_value=ok({"text": "thinking aloud", "thought": True}, text("Hello, "), text("Nova."))
    )
    reply = provider.chat([Message.user("hi")])
    assert reply.content == "Hello, Nova."
    assert reply.tool_calls == []


def test_a_call_keeps_the_models_id_and_gets_a_local_one_when_it_has_none(router, provider):
    router.post(GENERATE).mock(
        return_value=ok(
            call("memory_search", {"query": "a"}, id="fc-model"),
            call("memory_search", {"query": "b"}),
        )
    )
    first, second = provider.chat([Message.user("go")], [MEMORY_TOOL]).tool_calls
    assert (first.id, first.name, first.arguments) == ("fc-model", "memory_search", {"query": "a"})
    assert second.id.startswith(LOCAL_CALL_ID_PREFIX) and second.arguments == {"query": "b"}


def test_a_malformed_function_call_is_the_transient_response_class(router, provider):
    router.post(GENERATE).mock(return_value=ok(finish="MALFORMED_FUNCTION_CALL"))
    with pytest.raises(ProviderResponseError):
        provider.chat([Message.user("go")], [MEMORY_TOOL])


def test_a_blocked_reply_ends_the_turn_silent_and_says_why(router, provider, caplog):
    router.post(GENERATE).mock(return_value=ok(finish="SAFETY"))
    with caplog.at_level(logging.WARNING, logger="basecradle_harness"):
        reply = provider.chat([Message.user("go")])
    assert reply.content is None and reply.tool_calls == []
    assert "reason=SAFETY" in caplog.text


def test_the_output_budget_running_out_reads_as_truncated(router, provider):
    """Issue #490's capability: Gemini's ``MAX_TOKENS`` is one of the shared spellings of *length*."""
    router.post(GENERATE).mock(return_value=ok(text("half a sente"), finish="MAX_TOKENS"))
    provider.chat([Message.user("go")])
    assert provider.last_finish_reason == "MAX_TOKENS" and truncated(provider.last_finish_reason)


def test_the_providers_own_input_count_feeds_the_context_budget(router, provider):
    router.post(GENERATE).mock(return_value=ok(text("hi")))
    provider.chat([Message.user("go")])
    assert provider.last_tokens_in == 1200


# === the llm line ==============================================================================


def _llm_line(caplog) -> str:
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("llm ")]
    assert len(lines) == 1, lines
    return lines[0]


def test_the_llm_line_shape(router, provider, caplog):
    """The exact field order the NOC's columns key on (issue #655's completion comment quotes it).

    ``cost_basis=computed`` follows the computed ``cost=`` (issue #657), as the Steel line's does.
    """
    router.post(GENERATE).mock(return_value=ok(text("hi")))
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        provider.chat([Message.user("go")])
    line = _llm_line(caplog)
    assert re.fullmatch(
        r"llm provider=google purpose=main endpoint=us model=gemini-3\.8-flash "
        r"duration=\d+\.\d\ds tokens_in=1200 tokens_out=100 tokens_total=1300 cached_tokens=1000 "
        r"cost=\S+ cost_basis=computed generation_id=resp-0123456789abcdef",
        line,
    ), line


def test_the_cost_is_computed_from_googles_rates(router, provider, caplog, monkeypatch):
    """1,200 prompt (1,000 of it cached) + 100 out (30 reply + 70 thinking), non-global, intro rate.

    200 × $0.825/M + 1,000 × $0.0825/M + 100 × $4.125/M = $0.000660.
    """
    monkeypatch.setattr(
        "basecradle_harness._google_rates.billing_day", lambda: datetime.date(2026, 10, 6)
    )
    router.post(GENERATE).mock(return_value=ok(text("hi")))
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        provider.chat([Message.user("go")])
    assert " cost=0.00066 " in _llm_line(caplog)


def test_a_model_without_a_rate_gets_no_cost_and_one_warning(router, key_file, caplog):
    path = GENERATE.replace(MODEL, "gemini-9-experimental")
    router.post(path).mock(return_value=ok(text("hi")))
    with (
        GoogleProvider(
            "gemini-9-experimental", credentials_file=str(key_file), location=LOCATION
        ) as p,
        caplog.at_level(logging.INFO, logger="basecradle_harness"),
    ):
        p.chat([Message.user("go")])
        p.chat([Message.user("go again")])
    llm = [r.getMessage() for r in caplog.records if r.getMessage().startswith("llm ")]
    assert len(llm) == 2 and all("cost=" not in line for line in llm)
    warnings = [r for r in caplog.records if "No published rate" in r.getMessage()]
    assert len(warnings) == 1 and warnings[0].levelno == logging.WARNING


def test_a_priority_tier_call_is_not_priced_at_standard_rates(router, provider, caplog):
    usage = {
        "promptTokenCount": 100,
        "candidatesTokenCount": 10,
        "totalTokenCount": 110,
        "trafficType": "ON_DEMAND_PRIORITY",
    }
    router.post(GENERATE).mock(return_value=ok(text("hi"), usage=usage))
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        provider.chat([Message.user("go")])
    assert "cost=" not in _llm_line(caplog)


# === faults ====================================================================================


def _error(
    status: int, message: str, *, code: str = "INVALID_ARGUMENT", details=None, headers=None
):
    error = {"code": status, "message": message, "status": code}
    if details is not None:
        error["details"] = details
    return httpx.Response(status, json={"error": error}, headers=headers)


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (
            _error(
                400,
                "The input token count (1236488) exceeds the maximum number of tokens allowed (1048576).",
            ),
            ProviderContextLengthError,
        ),
        (
            _error(
                400,
                "Unable to submit request because the input token count is 131335 but model only supports up to 131072.",
            ),
            ProviderContextLengthError,
        ),
        (
            _error(400, "Request payload size exceeds the limit: 20971520 bytes."),
            ProviderPayloadTooLargeError,
        ),
        (_error(413, "Too large.", code="FAILED_PRECONDITION"), ProviderPayloadTooLargeError),
        (
            _error(401, "Request had invalid authentication credentials.", code="UNAUTHENTICATED"),
            ProviderAuthError,
        ),
        (
            _error(
                403, "Permission 'aiplatform.endpoints.predict' denied.", code="PERMISSION_DENIED"
            ),
            ProviderAuthError,
        ),
        (
            _error(
                403,
                "This API method requires billing to be enabled.",
                code="PERMISSION_DENIED",
                details=[
                    {
                        "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                        "reason": "BILLING_DISABLED",
                    }
                ],
            ),
            ProviderBillingError,
        ),
        (
            _error(
                403,
                "The billing account for the owning project is disabled in state absent",
                code="PERMISSION_DENIED",
            ),
            ProviderBillingError,
        ),
        (
            _error(429, "Resource exhausted. Please try again later.", code="RESOURCE_EXHAUSTED"),
            ProviderRateLimitError,
        ),
        (_error(500, "Internal error.", code="INTERNAL"), ProviderServerError),
        (
            _error(503, "The service is currently unavailable.", code="UNAVAILABLE"),
            ProviderServerError,
        ),
    ],
)
def test_vertex_faults_map_by_their_nature(router, provider, response, expected):
    router.post(GENERATE).mock(return_value=response)
    with pytest.raises(expected):
        provider.chat([Message.user("go")])


def test_a_generic_bad_request_propagates_as_a_plain_api_error(router, provider):
    """Never a reported class: a fixable config defect must leave the peer's message re-drivable."""
    router.post(GENERATE).mock(return_value=_error(400, "Invalid value at 'generation_config'."))
    with pytest.raises(ProviderAPIError) as raised:
        provider.chat([Message.user("go")])
    assert type(raised.value) is ProviderAPIError and raised.value.status_code == 400


def test_a_model_that_does_not_exist_is_a_404_api_error(router, provider):
    router.post(GENERATE).mock(
        return_value=_error(404, "Publisher Model was not found.", code="NOT_FOUND")
    )
    with pytest.raises(ProviderAPIError) as raised:
        provider.chat([Message.user("go")])
    assert raised.value.status_code == 404


def test_a_rate_limit_carries_the_vendors_retry_after(router, provider):
    router.post(GENERATE).mock(
        return_value=_error(
            429, "Quota exceeded.", code="RESOURCE_EXHAUSTED", headers={"Retry-After": "2"}
        )
    )
    with pytest.raises(ProviderRateLimitError) as raised:
        provider.chat([Message.user("go")])
    assert raised.value.retry_after == 2.0


@pytest.mark.parametrize(
    "response",
    [
        _error(504, "Deadline expired before operation could complete.", code="DEADLINE_EXCEEDED"),
        _error(500, "Deadline exceeded.", code="DEADLINE_EXCEEDED"),
    ],
)
def test_vertexs_own_deadline_is_a_timeout_not_a_server_fault(router, provider, response):
    """Retried once with twice the budget (#589), never re-sent into the same wall as a 5xx."""
    router.post(GENERATE).mock(return_value=response)
    with pytest.raises(ProviderTimeoutError):
        provider.chat([Message.user("go")])


def test_a_read_timeout_is_typed_a_timeout(router, provider):
    router.post(GENERATE).mock(side_effect=httpx.ReadTimeout("slow"))
    with pytest.raises(ProviderTimeoutError):
        provider.chat([Message.user("go")])


def test_a_dropped_connection_is_a_connection_error(router, provider):
    router.post(GENERATE).mock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(ProviderConnectionError) as raised:
        provider.chat([Message.user("go")])
    assert not isinstance(raised.value, ProviderTimeoutError)


def test_a_refused_key_is_an_auth_error(router, provider, monkeypatch):
    def refused(self, request):
        raise auth_errors.RefreshError("invalid_grant: Invalid JWT Signature.")

    monkeypatch.setattr(service_account.Credentials, "refresh", refused)
    with pytest.raises(ProviderAuthError):
        provider.chat([Message.user("go")])


def test_a_token_endpoint_blip_is_transient(router, provider, monkeypatch):
    def blip(self, request):
        raise auth_errors.RefreshError("temporarily unavailable", retryable=True)

    monkeypatch.setattr(service_account.Credentials, "refresh", blip)
    with pytest.raises(ProviderConnectionError):
        provider.chat([Message.user("go")])


# === the token refresh: bounded, single-shot =================================================


class _TokenResponse:
    def __init__(self, status: int) -> None:
        self.status = status
        self.headers = {}
        self.data = b'{"access_token": "ya29.x", "expires_in": 3600}'


def test_the_token_refresh_is_bounded_and_never_retried_inside_google_auth(monkeypatch):
    """One request, the fixed connect budget and a bounded read; a retryable status ends it at once."""
    from basecradle_harness._google import _token_request
    from basecradle_harness._timeouts import METADATA_TIMEOUT

    seen = []

    def answer(self, url, method="GET", body=None, headers=None, timeout=None, **kwargs):
        seen.append(timeout)
        return _TokenResponse(503 if len(seen) == 1 else 200)

    monkeypatch.setattr("google.auth.transport.requests.Request.__call__", answer)
    request = _token_request((CONNECT_TIMEOUT, METADATA_TIMEOUT))
    with pytest.raises(auth_errors.TransportError, match="503"):
        request("https://oauth2.googleapis.com/token", method="POST")
    with pytest.raises(auth_errors.TransportError, match="not retried"):
        request("https://oauth2.googleapis.com/token", method="POST")
    assert seen == [(CONNECT_TIMEOUT, METADATA_TIMEOUT)]


def test_an_expired_token_is_refreshed_by_the_adapter_on_its_own_terms(
    router, provider, monkeypatch
):
    from basecradle_harness._google import _Sealed

    transports = []

    def refresh(self, request):
        transports.append(type(request).__name__)
        self.token = FAKE_TOKEN
        self.expiry = datetime.datetime.now(datetime.timezone.utc).replace(
            tzinfo=None
        ) + datetime.timedelta(hours=1)

    monkeypatch.setattr(service_account.Credentials, "refresh", refresh)
    router.post(GENERATE).mock(return_value=ok(text("hi")))
    provider.chat([Message.user("hello")])
    provider.chat([Message.user("hello again")])
    # Refreshed once, by the adapter, through its one-shot transport — the SDK never had to.
    assert transports == ["_OneShot"]
    assert isinstance(provider._credentials, _Sealed)


# === capabilities ==============================================================================


def test_caching_is_automatic(provider):
    assert provider.cache_mode == AUTOMATIC


def test_a_gemini_model_sees_images_and_video(provider):
    assert provider.supports_vision() is True and provider.supports_video() is True


def test_a_model_outside_the_gemini_family_is_unknown(key_file):
    with GoogleProvider("text-model-x", credentials_file=str(key_file), location=LOCATION) as p:
        assert p.supports_vision() is None and p.supports_video() is None


def test_the_context_limit_is_unknown_and_costs_no_request(router, provider):
    """Vertex's model record, as this SDK reads it, carries no token limit — so nothing is asked.

    The budget then takes its floor, and ``HARNESS_MAX_CONTEXT_TOKENS`` is the operator's override.
    """
    assert provider.context_limit() is None
    assert not router.calls


def test_the_brain_describes_itself(provider):
    assert (provider.provider, provider.sdk, provider.surface, provider.model) == (
        "google",
        "google-genai",
        "native",
        MODEL,
    )


def _call(id: str, name: str, arguments: dict):
    from basecradle_harness import ToolCall

    return ToolCall(id=id, name=name, arguments=arguments)


# === the credential stays out of every representation ==========================================


def test_no_representation_of_the_adapter_emits_its_key_or_token(router, provider, key_file):
    """The issue #599 sweep, after a call has minted a token: neither the key nor the token shows.

    The adapter keeps no credential of its own — the key file is read once into the SDK's
    credentials object, behind the SDK's client — so this pins that no surface reaches through.
    """
    from tests.test_secret import SURFACES

    router.post(GENERATE).mock(return_value=ok(text("hi")))
    provider.chat([Message.user("hello")])
    key_line = json.loads(key_file.read_text())["private_key"].splitlines()[5]
    for name, surface in SURFACES.items():
        try:
            shown = surface(provider)
        except Exception:  # noqa: BLE001 - refusing is an acceptable answer; emitting is not
            shown = ""
        assert key_line not in shown and FAKE_TOKEN not in shown, name


# === Gemini's built-ins, and Google Search as a grounded call of its own (issue #656) =========
#
# Code execution and URL context ride beside the function declarations on every turn. Google Search
# cannot — Vertex refuses a search tool beside function calling — so `GoogleProvider.search` makes a
# separate call whose only tool is ``google_search``, and `GoogleSearchTool` hands its answer to the
# agent as ``web_search``. These pin what goes on the wire, the two lines a search writes (its
# ``llm`` line as a helper call, its grounding fee as a priced ``media`` line), and the honest gaps.

GROUNDED_USAGE = {
    "promptTokenCount": 20,
    "candidatesTokenCount": 120,
    "toolUsePromptTokenCount": 4000,
    "totalTokenCount": 4140,
    "trafficType": "ON_DEMAND",
}


def grounded(*parts, queries=("weather in chicago", "chicago forecast"), chunks=None, usage=None):
    """A grounded ``generateContent`` body: the answer, the queries Google ran, the sources."""
    payload = body(*parts, usage=usage or GROUNDED_USAGE)
    if chunks is None:
        chunks = [
            {"web": {"uri": "https://vertexaisearch.cloud.google.com/r/1", "title": "weather.gov"}},
            {
                "web": {
                    "uri": "https://vertexaisearch.cloud.google.com/r/2",
                    "title": "nws.noaa.gov",
                }
            },
        ]
    payload["candidates"][0]["groundingMetadata"] = {
        "webSearchQueries": list(queries),
        "groundingChunks": chunks,
    }
    return httpx.Response(200, json=payload)


def _lines_starting(caplog, head):
    return [m for m in (r.getMessage() for r in caplog.records) if m.startswith(head)]


@pytest.fixture
def _pinned_billing_day(monkeypatch):
    monkeypatch.setattr(
        "basecradle_harness._google_rates.billing_day", lambda: datetime.date(2026, 10, 7)
    )


# === built-ins beside the function declarations ================================================


def test_code_execution_and_url_context_ride_beside_the_function_declarations(router, key_file):
    provider = build(key_file, builtin_tools=["code_execution", "url_context"])
    route = router.post(GENERATE).mock(return_value=ok(text("hi")))
    provider.chat([Message.user("go")], tools=[MEMORY_TOOL])
    tools = sent(route)["tools"]
    assert tools[:2] == [{"codeExecution": {}}, {"urlContext": {}}]
    assert [d["name"] for d in tools[2]["functionDeclarations"]] == ["memory_search"]


def test_built_ins_are_sent_on_a_turn_with_no_function_tools(router, key_file):
    provider = build(key_file, builtin_tools=["code_execution"])
    route = router.post(GENERATE).mock(return_value=ok(text("hi")))
    provider.chat([Message.user("go")])
    assert sent(route)["tools"] == [{"codeExecution": {}}]


def test_google_search_is_never_sent_beside_the_function_declarations(router, key_file):
    """Vertex refuses that combination, so no builtin name ever puts it on a turn."""
    provider = build(key_file, builtin_tools=["google_search", "web_search", "url_context"])
    route = router.post(GENERATE).mock(return_value=ok(text("hi")))
    provider.chat([Message.user("go")], tools=[MEMORY_TOOL])
    wire = json.dumps(sent(route)["tools"])
    assert "googleSearch" not in wire and "google_search" not in wire
    assert sent(route)["tools"][0] == {"urlContext": {}}


def test_a_turn_with_neither_sends_no_tools_field(router, provider):
    route = router.post(GENERATE).mock(return_value=ok(text("hi")))
    provider.chat([Message.user("go")])
    assert "tools" not in sent(route)


def test_code_execution_parts_leave_the_models_own_words_as_the_reply(router, key_file):
    provider = build(key_file, builtin_tools=["code_execution"])
    parts = (
        text("Let me compute. "),
        {"executableCode": {"language": "PYTHON", "code": "print(2**10)"}},
        {"codeExecutionResult": {"outcome": "OUTCOME_OK", "output": "1024\n"}},
        text("It is 1024."),
    )
    router.post(GENERATE).mock(return_value=ok(*parts))
    reply = provider.chat([Message.user("2^10?")])
    assert reply.content == "Let me compute. It is 1024."


# === the grounded search call ==================================================================


def test_a_search_sends_google_search_alone_and_only_the_query(router, provider):
    route = router.post(GENERATE).mock(return_value=grounded(text("Sunny.")))
    provider.search("weather in chicago this weekend")
    request = sent(route)
    assert request["tools"] == [{"googleSearch": {}}]
    assert request["contents"] == [
        {"role": "user", "parts": [{"text": "weather in chicago this weekend"}]}
    ]
    assert "systemInstruction" not in request


def test_a_search_returns_the_grounded_answer_with_its_sources(router, provider):
    router.post(GENERATE).mock(return_value=grounded(text("Sunny, 55°F.")))
    answer = provider.search("weather in chicago")
    assert answer == (
        "Sunny, 55°F.\n\nSources:\n"
        "- weather.gov — https://vertexaisearch.cloud.google.com/r/1\n"
        "- nws.noaa.gov — https://vertexaisearch.cloud.google.com/r/2"
    )


def test_a_search_writes_a_helper_llm_line_and_a_priced_grounding_line(router, provider, caplog):
    router.post(GENERATE).mock(return_value=grounded(text("Sunny.")))
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        provider.search("weather in chicago")
    (llm,) = _lines_starting(caplog, "llm ")
    assert re.fullmatch(
        r"llm provider=google purpose=helper kind=search\.grounding endpoint=us "
        r"model=gemini-3\.8-flash duration=\d+\.\d\ds tokens_in=20 tokens_out=120 "
        r"tokens_total=4140 cost=\S+ cost_basis=computed \S*outcome=ok\S* "
        r"generation_id=resp-0123456789abcdef",
        llm,
    ), llm
    assert _lines_starting(caplog, "media ") == [
        (
            "media provider=google kind=search.grounding model=gemini-3.8-flash count=2 "
            "cost=0.028 cost_basis=computed"
        )
    ]


def test_grounding_tokens_are_not_charged_on_a_gemini_3_search(
    router, provider, caplog, _pinned_billing_day
):
    """ "Input tokens provided by Grounding with Google Search … are not charged" (Gemini 3)."""
    router.post(GENERATE).mock(return_value=grounded(text("Sunny.")))
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        provider.search("weather")
    cost = float(re.search(r" cost=([0-9.]+) ", _lines_starting(caplog, "llm ")[0]).group(1))
    # 20 in × $0.825/M + 120 out × $4.125/M, non-global, introductory rate; the 4,000 grounding
    # tokens nowhere.
    assert cost == pytest.approx((20 * 0.825 + 120 * 4.125) / 1e6, abs=1e-8)


def test_the_brains_own_turns_still_pay_for_tool_use_tokens(_pinned_billing_day):
    """Code execution's and URL context's tokens are billed as input: only grounding is exempt."""
    usage = Usage(prompt=20, cached=0, output=120, tool_prompt=4000)
    charged = call_cost(MODEL, "us", usage)
    exempt = call_cost(MODEL, "us", usage, grounded=True)
    assert charged - exempt == pytest.approx(4000 * 0.825 / 1e6)


def test_a_search_that_returned_no_sources_is_not_billed_a_grounding_fee(router, provider, caplog):
    router.post(GENERATE).mock(return_value=grounded(text("I could not find that."), chunks=[]))
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        answer = provider.search("something unfindable")
    assert answer == "I could not find that."
    assert not _lines_starting(caplog, "media ")


def test_a_fee_that_cannot_be_counted_is_logged_without_a_cost_and_warned(router, provider, caplog):
    router.post(GENERATE).mock(return_value=grounded(text("Sunny."), queries=()))
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        provider.search("weather")
    assert _lines_starting(caplog, "media ") == [
        "media provider=google kind=search.grounding model=gemini-3.8-flash"
    ]
    assert any(
        r.levelno == logging.WARNING and "grounding" in r.getMessage() for r in caplog.records
    )


def test_a_search_failure_is_the_same_typed_error_a_turn_raises(router, provider):
    router.post(GENERATE).mock(
        return_value=httpx.Response(
            429, json={"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}}
        )
    )
    with pytest.raises(ProviderError):
        provider.search("weather")


# === the web_search tool =======================================================================


class _Searcher:
    def __init__(self, answer="Sunny.", error=None):
        self.answer, self.error, self.queries = answer, error, []

    def search(self, query):
        self.queries.append(query)
        if self.error:
            raise self.error
        return self.answer


def test_the_tool_is_web_search_and_hands_the_model_the_grounded_answer():
    searcher = _Searcher()
    tool = GoogleSearchTool(searcher)
    assert tool.name == "web_search"
    assert tool.run(query="  weather in chicago  ") == "Sunny."
    assert searcher.queries == ["weather in chicago"]


def test_an_empty_query_is_refused_without_a_call():
    searcher = _Searcher()
    assert GoogleSearchTool(searcher).run(query="  ").startswith("Error:")
    assert searcher.queries == []


def test_a_provider_failure_reaches_the_model_as_text():
    tool = GoogleSearchTool(_Searcher(error=ProviderError("Vertex rate-limited the request")))
    assert tool.run(query="weather") == "Error searching the web: Vertex rate-limited the request"


def test_the_tool_searches_on_the_brains_own_vertex_configuration(router, key_file, monkeypatch):
    monkeypatch.setenv("AI_MODEL", MODEL)
    monkeypatch.setenv("AI_CREDENTIALS_FILE", str(key_file))
    monkeypatch.setenv("AI_LOCATION", "us")
    route = router.post(GENERATE).mock(return_value=grounded(text("Sunny.")))
    tool = GoogleSearchTool()
    assert tool.run(query="weather").startswith("Sunny.")
    assert tool.run(query="again").startswith("Sunny.")
    assert route.call_count == 2


def test_a_brain_with_no_vertex_configuration_gets_a_readable_error(monkeypatch):
    monkeypatch.setenv("AI_MODEL", MODEL)
    answer = GoogleSearchTool().run(query="weather")
    assert answer.startswith("Error searching the web: No Vertex location")


def test_a_brain_with_no_model_gets_a_readable_error(monkeypatch):
    monkeypatch.delenv("AI_MODEL", raising=False)
    assert "AI_MODEL" in GoogleSearchTool().run(query="weather")


def test_a_per_prompt_model_bills_one_grounding_unit_however_many_queries(router, key_file, caplog):
    """Gemini 2.5 bills the grounded prompt once ($35 per 1,000), so `count=` is 1, not the queries."""
    provider = build(key_file)
    provider.model = "gemini-2.5-flash"
    router.post(GENERATE.replace(MODEL, "gemini-2.5-flash")).mock(
        return_value=grounded(text("Sunny."), queries=("a", "b", "c"))
    )
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        provider.search("weather")
    assert _lines_starting(caplog, "media ") == [
        (
            "media provider=google kind=search.grounding model=gemini-2.5-flash count=1 "
            "cost=0.035 cost_basis=computed"
        )
    ]
