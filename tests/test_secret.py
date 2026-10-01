"""No representation of a credential-holding harness object emits the credential (issue #599).

The class (basecradle#612): *a generic serializer or representation of an object holding a
credential emits the credential.* Every object in this package that holds one is built here with a
fabricated, distinct fake, driven through every surface the sweep measured, and the output grepped
for every fake. A surface may refuse (raise); it may never emit. `json.dump` to a stream is checked
for what it *wrote before raising*, because a circular-reference error arrives after the bytes.

Each holder is also checked the other way round: the credential still reaches the one call that
needs it, revealed — a redaction that also stopped the key leaving would be a broken agent.
"""

from __future__ import annotations

import copy
import dataclasses
import io
import json
import logging
import pickle
import pprint
import subprocess

import httpx
import pytest
import respx

from basecradle_harness._audio import Transcriber
from basecradle_harness._direct_message import DirectMessageTool
from basecradle_harness._grok import GrokGenerateImageTool
from basecradle_harness._images import GenerateImageTool
from basecradle_harness._mcp import HttpMcpClient, McpServerConfig, StdioMcpClient, revealed
from basecradle_harness._openrouter_account import OpenRouterAccountBalanceTool
from basecradle_harness._platform import PlatformContext
from basecradle_harness._rerank import MemPalaceReranker
from basecradle_harness._secret import Secret, reveal, secret
from basecradle_harness._token import SelfHealingBaseCradle
from basecradle_harness._xai_account import XaiAccountBalanceTool
from basecradle_harness._xai_sdk import XaiSdkProvider
from tests.test_probe import NONCE, signed
from tests.test_token import NEW_TOKEN, minted, unauthorized
from tests.test_wake import BC_URL, TIMELINE_UUID, build_wake, platform  # noqa: F401 - fixture

# One fake per credential kind, each distinct, so a hit names which one leaked.
PLATFORM = "bc_uat_KqI8zFxkQ0OZ8vYwT7mWcVtR3nSdLpEa"
PASSWORD = "pw-Zx9-correct-horse-battery-staple"
OPENAI = "sk-proj-FAKEfakeFAKEfake0123456789abcdef"
XAI = "xai-FAKEfakeFAKEfake0123456789abcdefFAKE"
OPENROUTER = "sk-or-v1-fakefakefakefake0123456789abcdef0123"
MANAGEMENT = "mgmt-FAKEfakeFAKEfake0123456789"
NTFY = "tk_fakefakefakefake0123456789"
PROBE = "probe-secret-FAKE0123456789abcdef"
MCP_ENV = "ghp_FAKEfakeFAKEfake0123456789abcdef"
MCP_HEADER = "Bearer mcp-FAKEfakeFAKE0123456789"
FAKES = (PLATFORM, PASSWORD, OPENAI, XAI, OPENROUTER, MANAGEMENT, NTFY, PROBE, MCP_ENV, MCP_HEADER)


def _leaked(text: str) -> list[str]:
    return [fake for fake in FAKES if fake in text]


def _stream(obj: object) -> str:
    """What ``json.dump(obj, fp, default=vars)`` wrote — including before it raised."""
    fp = io.StringIO()
    try:
        json.dump(obj, fp, default=vars)
    except (TypeError, ValueError):
        pass
    return fp.getvalue()


def _logged(fmt: str, obj: object) -> str:
    stream = io.StringIO()
    logger = logging.getLogger("tests.test_secret")
    handler = logging.StreamHandler(stream)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        logger.debug(fmt, obj)
    finally:
        logger.removeHandler(handler)
    return stream.getvalue()


#: Every surface the sweep measured. Each returns the text it produced; one that raises refused.
SURFACES = {
    "repr": repr,
    "str": str,
    "format": lambda o: f"{o}",
    "pprint": pprint.pformat,
    "vars": lambda o: repr(vars(o)),
    # What Sentry, rich and cgitb render: every local's attributes, one level deep.
    "vars one level": lambda o: "\n".join(f"{k}={v!r}" for k, v in vars(o).items()),
    "json.dumps(default=vars)": lambda o: json.dumps(o, default=vars),
    "json.dump(fp, default=vars)": _stream,
    "json.dumps(vars, default=str)": lambda o: json.dumps(vars(o), default=str),
    "asdict": lambda o: repr(dataclasses.asdict(o)) if dataclasses.is_dataclass(o) else "",
    "pickle": lambda o: repr(pickle.dumps(o)),
    "__reduce_ex__": lambda o: repr(o.__reduce_ex__(2)),
    "copy": lambda o: repr(vars(copy.copy(o))),
    "deepcopy": lambda o: repr(vars(copy.deepcopy(o))),
    "logging %r": lambda o: _logged("%r", o),
    "logging %s": lambda o: _logged("%s", o),
}


def _assert_emits_nothing(obj: object) -> None:
    for name, surface in SURFACES.items():
        try:
            text = surface(obj)
        except Exception:  # noqa: BLE001 - refusing is an acceptable answer; emitting is not
            text = ""
        assert not _leaked(text), f"{type(obj).__name__} emits {_leaked(text)} through {name}"


def _mcp_config(**overrides) -> McpServerConfig:
    fields = {
        "name": "github",
        "command": "npx",
        "env": {"GITHUB_TOKEN": MCP_ENV},
        "headers": {"Authorization": MCP_HEADER},
    }
    return McpServerConfig(**{**fields, **overrides})


def _self_healing() -> SelfHealingBaseCradle:
    return SelfHealingBaseCradle(PLATFORM, email="nova@example.com", password=PASSWORD)


# --- the type itself -------------------------------------------------------------------------


def test_a_secret_shows_nothing_and_reveals_on_request():
    key = Secret(OPENAI)
    assert repr(key) == str(key) == f"{key}" == "Secret('[REDACTED]')"
    assert key.reveal() == OPENAI
    assert key == Secret(OPENAI) and key != Secret(XAI)
    assert bool(key) and not Secret("")


def test_a_secret_has_no_dict_to_expand():
    with pytest.raises(TypeError):
        vars(Secret(OPENAI))


def test_a_secret_refuses_pickle_naming_the_risk():
    with pytest.raises(TypeError, match="holds a credential"):
        pickle.dumps(Secret(OPENAI))
    with pytest.raises(TypeError, match="holds a credential"):
        Secret(OPENAI).__reduce_ex__(2)


def test_a_copy_is_the_same_secret_never_a_second_one():
    key = Secret(OPENAI)
    assert copy.copy(key) is key
    assert copy.deepcopy({"k": key})["k"] is key


def test_the_optional_helpers_keep_none_meaning_read_the_environment():
    assert secret(None) is None and secret("") is None
    assert reveal(None) is None
    assert reveal(secret(OPENAI)) == OPENAI


# --- every holder, every surface --------------------------------------------------------------


HOLDERS = {
    "SelfHealingBaseCradle": _self_healing,
    "PlatformContext": lambda: PlatformContext(client=_self_healing(), timeline=TIMELINE_UUID),
    "XaiSdkProvider": lambda: XaiSdkProvider("grok-4.3", api_key=XAI),
    "MemPalaceReranker": lambda: MemPalaceReranker(model="m", api_key=OPENROUTER, providers=["x"]),
    "GenerateImageTool": lambda: GenerateImageTool(api_key=OPENAI),
    "GrokGenerateImageTool": lambda: GrokGenerateImageTool(api_key=XAI),
    "Transcriber": lambda: Transcriber(api_key=OPENAI),
    "XaiAccountBalanceTool": lambda: XaiAccountBalanceTool(management_key=MANAGEMENT),
    "OpenRouterAccountBalanceTool": lambda: OpenRouterAccountBalanceTool(management_key=MANAGEMENT),
    "DirectMessageTool": lambda: DirectMessageTool(token=NTFY),
    "McpServerConfig": _mcp_config,
    "StdioMcpClient": lambda: StdioMcpClient(_mcp_config(), 5.0),
    "HttpMcpClient": lambda: HttpMcpClient(_mcp_config(command=None, url="https://h/mcp"), 5.0),
}


@pytest.mark.parametrize("build", HOLDERS.values(), ids=HOLDERS.keys())
def test_no_representation_of_a_holder_emits_its_credential(build):
    _assert_emits_nothing(build())


def test_no_representation_of_the_wake_agent_emits_its_probe_secret(platform, tmp_path):  # noqa: F811
    agent, _ = build_wake(tmp_path, probe_secret=PROBE)
    _assert_emits_nothing(agent)


def test_the_platform_client_floor_carries_the_sdk_fix():
    """`basecradle>=0.13.0` is this repo's half of basecradle-python#242 — pinned, so a floor that
    slips back to a release whose client kept its token in ``__dict__`` fails here."""
    _assert_emits_nothing(SelfHealingBaseCradle(PLATFORM))


def test_an_mcp_config_still_shows_which_variables_it_sets():
    shown = repr(_mcp_config())
    assert "GITHUB_TOKEN" in shown and "Authorization" in shown
    assert "Secret('[REDACTED]')" in shown


# --- …and the credential still reaches the call that needs it -------------------------------


def test_a_remint_sends_the_real_password(tmp_path):
    with respx.mock(base_url=BC_URL) as router:
        router.get("/ping").mock(side_effect=[unauthorized(), httpx.Response(200, json={})])
        login = router.post("/session").mock(return_value=minted())
        client = _self_healing()
        try:
            client.request("GET", "/ping")
        finally:
            client.close()
    assert PASSWORD in login.calls.last.request.content.decode()
    assert client.token == NEW_TOKEN


def test_a_probe_still_verifies_against_the_real_secret(platform, tmp_path):  # noqa: F811
    agent, _ = build_wake(tmp_path, probe_secret=PROBE)
    assert agent._probe_nonce(signed(NONCE, PROBE)) == NONCE


def test_a_rebuilt_xai_client_gets_the_key_itself(monkeypatch):
    provider = XaiSdkProvider("grok-4.3", api_key=XAI)
    built: list[dict] = []
    monkeypatch.setattr(provider._xai, "Client", lambda **kwargs: built.append(kwargs) or object())
    provider.bind_conversation("timeline:019e7750-66ee-7f53-829f-13a8a710b6da")
    provider._bound_client()
    assert built and built[-1]["api_key"] == XAI


def test_an_http_mcp_client_sends_the_header_itself():
    client = HttpMcpClient(_mcp_config(command=None, url="https://h/mcp"), 5.0)
    assert client._client.headers["Authorization"] == MCP_HEADER
    assert revealed(_mcp_config().headers) == {"Authorization": MCP_HEADER}


def test_a_stdio_mcp_server_is_started_with_the_env_value_itself(monkeypatch):
    seen: dict = {}

    def popen(*args, **kwargs):
        seen.update(kwargs)
        raise OSError("not started in this test")

    monkeypatch.setattr(subprocess, "Popen", popen)
    with pytest.raises(Exception):  # noqa: B017 - the start failing is the point; we read its env
        StdioMcpClient(_mcp_config(), 5.0).start()
    assert seen["env"]["GITHUB_TOKEN"] == MCP_ENV


@pytest.mark.parametrize(
    ("build", "attribute", "value"),
    [
        (lambda: GenerateImageTool(api_key=OPENAI), "_key", OPENAI),
        (lambda: GrokGenerateImageTool(api_key=XAI), "_key", XAI),
        (lambda: Transcriber(api_key=OPENAI), "key", OPENAI),
    ],
    ids=["GenerateImageTool", "GrokGenerateImageTool", "Transcriber"],
)
def test_a_tool_still_hands_its_explicit_key_on(build, attribute, value):
    found = getattr(build(), attribute)
    assert (found() if callable(found) else found) == value
