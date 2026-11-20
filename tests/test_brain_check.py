"""The brain check sends a real wake's request, once or twice, and says what the vendor did (issue #947).

An agent moved to refuseless's ``kimi-k3.1-pro-abliterated`` failed its first real wake with a 400
(issue #945), after a hand-made call that was not shaped like a wake had passed. What is pinned here:

- the request is the wake's, read off the bytes the real ``openai`` SDK sent (respx at the
  transport): the brief before the fixed user turn, the step note after it, every resolved tool;
- a tool call is answered with the fixed result and the second step is sent, and never a third;
- #945's regression against a vendor double that refuses the shape the real one refuses: refused
  with its 400 when ``AI_SYSTEM_TURNS`` is unset, ok with ``before_user``;
- every verdict and its exit code, a candidate ``model_params.json``, and that nothing is written,
  no tool runs and neither the key nor the prompt reaches the output.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import httpx
import pytest
import respx

from basecradle_harness import _brain_check
from basecradle_harness._basecradle import _config_from_env, _resolve_tools
from basecradle_harness._brain_check import (
    STAND_IN_RESULT,
    USER_TURN,
    StandIn,
    check,
    main,
)
from basecradle_harness._exceptions import (
    ProviderError,
    ProviderResponseError,
    ProviderTimeoutError,
)
from basecradle_harness._observability import tally_llm_calls
from basecradle_harness._openai import OpenAIProvider
from basecradle_harness._system_turns import SYSTEM_LABEL

from .conftest import completion, wire_tool_call

VENDOR = "refuseless"
HOST = "https://api.refuseless.com/v1"
MODEL = "kimi-k3.1-pro-abliterated"
FAKE_KEY = "abl_0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKL"
PRICING = {"currency": "USD", "input_per_m": 0.6, "cached_input_per_m": 0.15, "output_per_m": 2.5}
USAGE = {"prompt_tokens": 1200, "completion_tokens": 40, "total_tokens": 1240}
#: The vendor's refusal, as the capital read it on 2026-11-20 (issue #945).
REFUSAL = {
    "error": {
        "message": "Invalid or unsupported request parameters",
        "type": "invalid_request",
        "code": "invalid_request",
    }
}


@pytest.fixture
def agent(monkeypatch, tmp_path):
    """A refuseless agent on a fresh box: a config home with nothing installed, and a home."""
    for name in ("config", "home"):
        (tmp_path / name).mkdir()
    monkeypatch.setenv("BASECRADLE_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("HARNESS_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("AI_PROVIDER", VENDOR)
    monkeypatch.setenv("AI_MODEL", MODEL)
    monkeypatch.setenv("AI_API_KEY", FAKE_KEY)
    monkeypatch.setenv("HARNESS_RESPONSE_RETRIES", "0")
    for var in (
        "AI_SDK",
        "AI_SDK_SURFACE",
        "AI_BASE_URL",
        "AI_SYSTEM_TURNS",
        "HARNESS_ONBOARD",
        "HARNESS_MAX_STEPS",
        "BASECRADLE_TOKEN",
        "BASECRADLE_TIMELINE",
    ):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


def _reply(**fields) -> dict:
    body = completion(**fields)
    body["model"] = MODEL
    body["usage"] = USAGE
    return body


def _tool_reply(name: str = "memory", call_id: str = "call_0001") -> dict:
    call = wire_tool_call(id=call_id, name=name, arguments={"action": "list"})
    return _reply(tool_calls=[call], finish_reason="tool_calls")


def _orphaned(messages: list[dict]) -> bool:
    """The real vendor's rule (issue #945): a system message no user turn follows is refused."""
    roles = [m["role"] for m in messages]
    for index, role in enumerate(roles):
        if role == "system":
            rest = [r for r in roles[index:] if r != "system"]
            if not rest or rest[0] != "user":
                return True
    return False


def _vendor(router, replies: list[dict], *, strict: bool = False) -> list[dict]:
    """Mock the vendor: its models list, and one scripted reply per request. Returns what was sent."""
    row = {"id": MODEL, "object": "model", "created": 0, "owned_by": VENDOR, "pricing": PRICING}
    router.get(f"{HOST}/models").mock(
        return_value=httpx.Response(200, json={"object": "list", "data": [row]})
    )
    sent: list[dict] = []

    def answer(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        sent.append(body)
        if strict and _orphaned(body["messages"]):
            return httpx.Response(400, json=REFUSAL)
        return httpx.Response(200, json=replies[len(sent) - 1])

    router.post(f"{HOST}/chat/completions").mock(side_effect=answer)
    return sent


def _run(capsys, argv: list[str] | None = None) -> tuple[int, dict, str]:
    code = main(argv or [])
    out = capsys.readouterr().out
    return code, json.loads(out), out


def _roles(body: dict) -> str:
    return "".join(
        {"system": "S", "user": "U", "assistant": "A", "tool": "T"}[m["role"]]
        for m in body["messages"]
    )


# --- the request is a wake's ----------------------------------------------------------------------


def test_a_tool_call_takes_the_check_through_the_second_step_of_a_wake(agent, capsys):
    with respx.mock(assert_all_mocked=True) as router:
        sent = _vendor(router, [_tool_reply(), _reply(content="Hello.")])
        code, verdict, _ = _run(capsys)

    assert (code, verdict["verdict"]) == (0, "ok")
    assert (verdict["rounds"], verdict["tool_called"]) == (2, True)
    first, second = sent
    # The brief, the newest message, the step note: the order a wake's first step sends.
    assert _roles(first) == "SUS"
    brief, user, note = first["messages"]
    assert "<initialize.md>" in brief["content"]
    assert user["content"] == USER_TURN
    assert note["content"].startswith("Current Time: ")
    assert "Step 1 of 24" in note["content"]
    # The second step carries the first step's note before the call it preceded, and its own last.
    assert _roles(second) == "SUSATS"
    assert second["messages"][4] == {
        "role": "tool",
        "tool_call_id": "call_0001",
        "content": STAND_IN_RESULT,
    }
    assert "Step 2 of 24" in second["messages"][-1]["content"]


def test_every_tool_the_wake_resolves_is_offered_by_its_own_definition(agent, capsys):
    provider, sdk, surface = _config_from_env()
    resolved, _ = _resolve_tools(provider, sdk, surface)
    with respx.mock(assert_all_mocked=True) as router:
        sent = _vendor(router, [_reply(content="Hello.")])
        _, verdict, _ = _run(capsys)

    offered = {tool["function"]["name"]: tool["function"] for tool in sent[0]["tools"]}
    assert set(offered) == {tool.name for tool in resolved.tools}
    assert verdict["tools_offered"] == len(offered)
    for tool in resolved.tools:
        spec = tool.to_spec()
        assert offered[spec.name]["description"] == spec.description
        assert offered[spec.name]["parameters"] == spec.parameters


def test_a_reply_that_ends_the_turn_ends_the_check_at_one_round(agent, capsys):
    with respx.mock(assert_all_mocked=True) as router:
        sent = _vendor(router, [_reply(content="Hello.")])
        code, verdict, _ = _run(capsys)
    assert (code, len(sent), verdict["rounds"], verdict["tool_called"]) == (0, 1, 1, False)


def test_a_third_request_is_never_sent(agent, capsys):
    """A model that keeps calling tools is stopped after its second reply, before a third call."""
    with respx.mock(assert_all_mocked=True) as router:
        sent = _vendor(router, [_tool_reply(), _tool_reply(call_id="call_0002"), _reply()])
        code, verdict, _ = _run(capsys)
    assert (code, verdict["verdict"], len(sent), verdict["rounds"]) == (0, "ok", 2, 2)


def test_without_onboarding_the_request_carries_no_brief(agent, capsys, monkeypatch):
    monkeypatch.setenv("HARNESS_ONBOARD", "0")
    with respx.mock(assert_all_mocked=True) as router:
        sent = _vendor(router, [_reply(content="Hello.")])
        _run(capsys)
    assert _roles(sent[0]) == "US"


# --- issue #945's regression ----------------------------------------------------------------------


def test_the_945_shape_is_refused_with_the_vendors_400_when_nothing_is_declared(agent, capsys):
    with respx.mock(assert_all_mocked=True, assert_all_called=False) as router:
        sent = _vendor(router, [_reply(content="Hello.")], strict=True)
        code, verdict, out = _run(capsys)

    assert (code, verdict["verdict"], verdict["status"]) == (70, "refused", 400)
    assert "Invalid or unsupported request parameters" in verdict["error"]
    assert verdict["system_turns"] == "anywhere"
    assert (len(sent), verdict["rounds"]) == (1, 0)
    assert FAKE_KEY not in out and USER_TURN not in out


def test_the_945_shape_is_accepted_when_before_user_is_declared(agent, capsys, monkeypatch):
    monkeypatch.setenv("AI_SYSTEM_TURNS", "before_user")
    with respx.mock(assert_all_mocked=True) as router:
        sent = _vendor(router, [_tool_reply(), _reply(content="Hello.")], strict=True)
        code, verdict, _ = _run(capsys)

    assert (code, verdict["verdict"], verdict["system_turns"]) == (0, "ok", "before_user")
    assert (_roles(sent[0]), _roles(sent[1])) == ("SUU", "SUUATU")
    assert sent[1]["messages"][-1]["content"].startswith(SYSTEM_LABEL + "Current Time: ")


def test_a_refusal_on_the_second_step_is_reported_with_the_round_it_reached(agent, capsys):
    """The order #945's fix had to cover: step 1 accepted, step 2 refused."""
    with respx.mock(assert_all_mocked=True) as router:
        router.get(f"{HOST}/models").mock(
            return_value=httpx.Response(200, json={"object": "list", "data": []})
        )
        router.post(f"{HOST}/chat/completions").mock(
            side_effect=[
                httpx.Response(200, json=_tool_reply()),
                httpx.Response(400, json=REFUSAL),
            ]
        )
        code, verdict, _ = _run(capsys)
    assert (code, verdict["verdict"], verdict["status"], verdict["rounds"]) == (
        70,
        "refused",
        400,
        1,
    )


# --- the verdict ----------------------------------------------------------------------------------


def test_ok_reports_each_calls_usage_and_their_sum(agent, capsys):
    with respx.mock(assert_all_mocked=True) as router:
        _vendor(router, [_tool_reply(), _reply(content="Hello.")])
        _, verdict, _ = _run(capsys)

    calls = verdict["calls"]
    assert [(c["step"], c["outcome"], c["purpose"]) for c in calls] == [
        ("1/24", "ok", "main"),
        ("2/24", "ok", "main"),
    ]
    one = (1200 * 0.6 + 40 * 2.5) / 1_000_000
    for call in calls:
        assert (call["tokens_in"], call["tokens_out"]) == (1200, 40)
        assert call["cost"] == pytest.approx(one)
    usage = verdict["usage"]
    assert (usage["calls"], usage["tokens_in"], usage["tokens_out"]) == (2, 2400, 80)
    assert usage["cost"] == pytest.approx(2 * one)
    assert {k: verdict[k] for k in ("provider", "sdk", "surface", "model")} == {
        "provider": VENDOR,
        "sdk": "openai",
        "surface": "chat",
        "model": MODEL,
    }


def test_no_answer_at_all_is_failed_never_refused(agent, capsys):
    with respx.mock(assert_all_mocked=True, assert_all_called=False) as router:
        router.get(f"{HOST}/models").mock(side_effect=httpx.ConnectError("refused"))
        router.post(f"{HOST}/chat/completions").mock(side_effect=httpx.ConnectError("no route"))
        code, verdict, _ = _run(capsys)
    assert (code, verdict["verdict"], verdict["rounds"]) == (69, "failed", 0)
    assert "status" not in verdict


@pytest.mark.parametrize(
    ("var", "value", "says"),
    [
        ("AI_PROVIDER", "Not A Slug", "AI_PROVIDER"),
        ("AI_SYSTEM_TURNS", "trailing_user", "AI_SYSTEM_TURNS"),
        ("AI_MODEL", "", "AI_MODEL is required"),
    ],
)
def test_a_configuration_a_wake_refuses_is_a_config_error(
    agent, capsys, monkeypatch, var, value, says
):
    monkeypatch.setenv(var, value)
    with respx.mock(assert_all_mocked=True) as router:
        code, verdict, _ = _run(capsys)
        assert not router.calls
    assert (code, verdict["verdict"]) == (64, "config_error")
    assert says in verdict["error"]


def test_a_vendor_error_echoing_the_key_is_scrubbed(agent, capsys):
    echoed = {
        "error": {"message": f"Incorrect API key provided: {FAKE_KEY}", "type": "invalid_request"}
    }
    with respx.mock(assert_all_mocked=True, assert_all_called=False) as router:
        router.get(f"{HOST}/models").mock(
            return_value=httpx.Response(200, json={"object": "list", "data": []})
        )
        router.post(f"{HOST}/chat/completions").mock(return_value=httpx.Response(400, json=echoed))
        code, verdict, out = _run(capsys)
    assert (code, verdict["status"]) == (70, 400)
    assert FAKE_KEY not in out
    assert "Incorrect API key provided" in verdict["error"]


# --- a candidate model_params.json ----------------------------------------------------------------


def test_a_candidate_model_params_is_what_the_request_carries(agent, capsys):
    (agent / "config" / "model_params.json").write_text(json.dumps({"temperature": 0.9}))
    candidate = agent / "candidate.json"
    candidate.write_text(json.dumps({"temperature": 0.2}))
    with respx.mock(assert_all_mocked=True) as router:
        sent = _vendor(router, [_reply(content="Hello.")])
        code, verdict, _ = _run(capsys, ["--model-params", str(candidate)])
    assert (code, sent[0]["temperature"], verdict["model_params"]) == (0, 0.2, str(candidate))


def test_the_installed_model_params_is_read_when_no_candidate_is_named(agent, capsys):
    (agent / "config" / "model_params.json").write_text(json.dumps({"temperature": 0.9}))
    with respx.mock(assert_all_mocked=True) as router:
        sent = _vendor(router, [_reply(content="Hello.")])
        _, verdict, _ = _run(capsys)
    assert (sent[0]["temperature"], verdict["model_params"]) == (0.9, None)


@pytest.mark.parametrize(("content", "says"), [(None, "does not exist"), ("[1]", "JSON object")])
def test_a_candidate_that_will_not_load_is_a_config_error(agent, capsys, content, says):
    candidate = agent / "candidate.json"
    if content is not None:
        candidate.write_text(content)
    with respx.mock(assert_all_mocked=True) as router:
        code, verdict, _ = _run(capsys, ["--model-params", str(candidate)])
        assert not router.calls
    assert (code, verdict["verdict"]) == (64, "config_error")
    assert str(candidate) in verdict["error"] and says in verdict["error"]


# --- it touches nothing ---------------------------------------------------------------------------


def _tree(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


def test_nothing_is_written_and_no_platform_is_reached(agent, capsys):
    """`assert_all_mocked` fails any request but the vendor's, the platform's included."""
    (agent / "config" / "model_params.json").write_text("{}")
    before = _tree(agent)
    with respx.mock(assert_all_mocked=True) as router:
        _vendor(router, [_tool_reply("messages"), _reply(content="Hello.")])
        code, _, _ = _run(capsys)
        hosts = {call.request.url.host for call in router.calls}
    assert code == 0
    assert hosts == {"api.refuseless.com"}
    assert _tree(agent) == before


def test_a_stand_in_offers_the_real_definition_and_never_runs_the_tool():
    class Real(_brain_check.Tool):
        name = "post_it"
        description = "Posts it."
        parameters = {"type": "object", "properties": {"text": {"type": "string"}}}
        requires = frozenset({"network"})

        def run(self, **kwargs):
            raise AssertionError("the real tool ran")

    real = Real()
    stand_in = StandIn(real)
    assert stand_in.to_spec() == real.to_spec()
    assert stand_in.requires == real.requires
    assert stand_in.run(text="hello") == STAND_IN_RESULT


def test_the_tally_is_read_off_the_line_and_changes_nothing_outside_its_block(caplog):
    from basecradle_harness._observability import log_llm_call

    caplog.set_level(logging.INFO, logger="basecradle_harness")
    with tally_llm_calls() as calls:
        log_llm_call(
            provider=VENDOR,
            model=MODEL,
            seconds=1.0,
            usage={"prompt_tokens": 10, "completion_tokens": 2},
            cost=0.000123,
            generation_id="gen-0001",
        )
    log_llm_call(provider=VENDOR, model=MODEL, seconds=1.0)
    assert calls == [
        {
            "purpose": "main",
            "outcome": "ok",
            "tokens_in": 10,
            "tokens_out": 2,
            "tokens_total": 12,
            "cost": 0.000123,
            "generation_id": "gen-0001",
        }
    ]
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("llm ")]
    assert len(lines) == 2 and "cost=0.000123" in lines[0]


def test_check_returns_the_verdict_main_prints(agent):
    with respx.mock(assert_all_mocked=True) as router:
        _vendor(router, [_reply(content="Hello.")])
        verdict = check()
    assert verdict["verdict"] == "ok"


@pytest.mark.parametrize(
    ("raised", "verdict", "code"),
    [
        # OpenRouter's typed SDK refusing a model_params.json key: a request never sent.
        (ProviderError("A key in model_params.json was not accepted"), "config_error", 64),
        (ProviderResponseError("EOF while parsing a value"), "failed", 69),
        (ProviderTimeoutError("did not answer in time"), "failed", 69),
    ],
)
def test_what_a_fault_is_decides_its_verdict(agent, capsys, monkeypatch, raised, verdict, code):
    def chat(self, messages, tools=None):
        raise raised

    monkeypatch.setattr(OpenAIProvider, "chat", chat)
    got, report, _ = _run(capsys)
    assert (got, report["verdict"]) == (code, verdict)
    assert str(raised) in report["error"]
