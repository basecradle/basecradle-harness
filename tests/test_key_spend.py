"""The `key spend` line: the brain's key's billed spend, once per wake — issue #868.

All HTTP is mocked with respx at the transport; no test reaches the network. The refuseless body is
the live ``GET /v1/credits`` shape the capital read on 2026-11-10; the OpenRouter body is the
``GET /key`` shape `_openrouter_account` documents (abridged).

What these pin, each in its "obviously fine" broken form:

- **The figure is the vendor's, to its own precision, on `spent=` and never `cost=`** — OpenRouter's
  ``usage`` unrounded, where the balance tool shows cents.
- **Only a refuseless or OpenRouter brain reads, and only when the wired brain and `AI_PROVIDER`
  agree**: any other brain, a test double, or a stale environment sends nothing anywhere.
- **No figure is a WARNING with its slug, never silence and never a raise**, and nothing but the
  slug: not the key, not the URL, not the vendor's prose.
- **The wake writes it once, just before `wake end`, and a stopped wake writes none.**
"""

from __future__ import annotations

import json
import logging

import httpx
import pytest
import respx

from basecradle_harness import _deadline
from basecradle_harness._key_spend import log_key_spend
from basecradle_harness._stop import WakeStopped
from tests.conftest import plain
from tests.test_wake import (
    BC_URL,
    M0,
    REPLY,
    TIMELINE_UUID,
    asset_page,
    build_wake,
    dashboard,
    event_page,
    message,
    page,
    serve_messages,
    task_page,
    timeline,
)

KEY = "sk-rl-0123456789abcdef0123456789abcdef"
REFUSELESS = "https://api.refuseless.com/v1/credits"
OPENROUTER = "https://openrouter.ai/api/v1/key"
TIMELINE = "019f6e71-2a12-7b69-a204-0fec1497b9c2"
DELIVERY = "019f6e71-2a12-7b69-a204-0fec1497b9c3"
#: A sentinel in a vendor's error prose, which must never reach the line.
PROSE = "account 019f6e71-2a12-7b69-a204-0fec1497b9c4 belongs to john@example.com"


def _credits(**overrides) -> dict:
    body = {
        "object": "credits",
        "balance": 69.989469,
        "reserved": 0,
        "available": 69.989469,
        "key_limit": None,
        "key_spent": 0.001405,
        "key_spent_since": None,
        "unit": "usd",
        "updated_at": 1794353525,
    }
    for field, value in overrides.items():
        if value is ...:
            body.pop(field)
        else:
            body[field] = value
    return body


def _key(**overrides) -> dict:
    data = {"usage": 1.047832516, "limit": None, "limit_remaining": None, "limit_reset": None}
    data.update(overrides)
    return {"data": {k: v for k, v in data.items() if v is not ...}}


def _ok(body: dict) -> httpx.Response:
    """A 200 as Python's `json` writes it, so ``NaN`` and ``Infinity`` reach the parser."""
    return httpx.Response(
        200, content=json.dumps(body).encode(), headers={"content-type": "application/json"}
    )


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "refuseless")
    monkeypatch.setenv("AI_API_KEY", KEY)
    monkeypatch.delenv("AI_BASE_URL", raising=False)


def _lines(caplog) -> list[tuple[str, str]]:
    return [
        (r.levelname, plain(r.getMessage()))
        for r in caplog.records
        if r.getMessage().startswith("key spend")
    ]


def _log(caplog, brain="refuseless", url=REFUSELESS, response=None) -> list[tuple[str, str]]:
    with respx.mock(assert_all_mocked=True, assert_all_called=False) as mock:
        route = mock.get(url)
        if isinstance(response, httpx.Response):
            route.mock(return_value=response)
        elif response is not None:
            route.mock(side_effect=response)
        with caplog.at_level(logging.INFO, logger="basecradle_harness"):
            log_key_spend(brain, timeline=TIMELINE, delivery=DELIVERY)
    return _lines(caplog)


# --- the figure ---------------------------------------------------------------------------------


def test_a_refuseless_brain_logs_the_vendors_figure_and_read_time(caplog):
    lines = _log(caplog, response=httpx.Response(200, json=_credits()))
    assert lines == [
        (
            "INFO",
            (
                f"key spend provider=refuseless outcome=ok spent=0.001405 "
                f"as_of=2026-11-10T23:32:05Z timeline={TIMELINE} delivery={DELIVERY}"
            ),
        )
    ]


def test_a_counter_start_rides_the_line(caplog):
    body = _credits(key_spent=12.5, key_spent_since=1794268661)
    ((_, line),) = _log(caplog, response=httpx.Response(200, json=body))
    assert " spent=12.5 since=2026-11-09T23:57:41Z as_of=2026-11-10T23:32:05Z " in line


@pytest.mark.parametrize("spent", [0, 0.0, -0.0])
def test_a_key_that_has_spent_nothing_says_zero(caplog, spent):
    ((level, line),) = _log(caplog, response=_ok(_credits(key_spent=spent)))
    assert level == "INFO" and " spent=0 " in line


@pytest.mark.parametrize(
    ("spent", "text"),
    [(0.0000001, "0.0000001"), (1234567.123456, "1234567.123456"), (7, "7"), (0.1, "0.1")],
)
def test_the_figure_is_every_digit_the_vendor_stated_and_never_an_exponent(caplog, spent, text):
    ((_, line),) = _log(caplog, response=_ok(_credits(key_spent=spent)))
    assert f" spent={text} " in line


def test_an_openrouter_brain_logs_the_vendors_precision_not_the_tools_cents(monkeypatch, caplog):
    monkeypatch.setenv("AI_PROVIDER", "openrouter")
    lines = _log(caplog, "openrouter", OPENROUTER, httpx.Response(200, json=_key()))
    assert lines == [
        (
            "INFO",
            (
                "key spend provider=openrouter outcome=ok spent=1.047832516 "
                f"timeline={TIMELINE} delivery={DELIVERY}"
            ),
        )
    ]


def test_the_figure_is_never_a_cost(caplog):
    ((_, line),) = _log(caplog, response=httpx.Response(200, json=_credits()))
    assert "cost=" not in line


# --- where it reads, and where it never does ----------------------------------------------------


@pytest.mark.parametrize("vendor", ["openai", "xai", "anthropic", "google", "wiro"])
def test_any_other_brain_reads_nothing_and_logs_nothing(monkeypatch, caplog, vendor):
    monkeypatch.setenv("AI_PROVIDER", vendor)
    with (
        respx.mock(assert_all_mocked=True) as mock,
        caplog.at_level(logging.INFO, logger="basecradle_harness"),
    ):
        log_key_spend(vendor, timeline=TIMELINE, delivery=None)
    assert not mock.calls
    assert _lines(caplog) == []


@pytest.mark.parametrize(
    ("env", "brain"),
    [("refuseless", "unknown"), ("refuseless", "openrouter"), ("openrouter", "refuseless")],
)
def test_the_wired_brain_and_the_environment_must_agree(monkeypatch, caplog, env, brain):
    """A test double, or a brain built beside a stale `AI_PROVIDER`, sends the key nowhere."""
    monkeypatch.setenv("AI_PROVIDER", env)
    with (
        respx.mock(assert_all_mocked=True) as mock,
        caplog.at_level(logging.INFO, logger="basecradle_harness"),
    ):
        log_key_spend(brain, timeline=TIMELINE, delivery=None)
    assert not mock.calls
    assert _lines(caplog) == []


def test_the_read_goes_where_the_brains_calls_go(monkeypatch, caplog):
    monkeypatch.setenv("AI_BASE_URL", "https://eu.refuseless.example.com/v1/")
    url = "https://eu.refuseless.example.com/v1/credits"
    with respx.mock(assert_all_mocked=True) as mock:
        route = mock.get(url).mock(return_value=httpx.Response(200, json=_credits()))
        with caplog.at_level(logging.INFO, logger="basecradle_harness"):
            log_key_spend("refuseless", timeline=TIMELINE, delivery=None)
    assert route.calls.last.request.headers["authorization"] == f"Bearer {KEY}"
    assert "outcome=ok" in _lines(caplog)[0][1]


def test_no_key_is_a_config_fault_and_sends_nothing(monkeypatch, caplog):
    monkeypatch.delenv("AI_API_KEY")
    with (
        respx.mock(assert_all_mocked=True) as mock,
        caplog.at_level(logging.INFO, logger="basecradle_harness"),
    ):
        log_key_spend("refuseless", timeline=TIMELINE, delivery=None)
    assert not mock.calls
    assert _lines(caplog) == [
        (
            "WARNING",
            (
                f"key spend provider=refuseless outcome=error reason=config:missing_api_key "
                f"timeline={TIMELINE}"
            ),
        )
    ]


# --- no figure is a warning with its slug, and nothing else ------------------------------------


def _error(status: int) -> httpx.Response:
    return httpx.Response(status, json={"error": {"code": "x", "message": PROSE, "type": "x"}})


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        (_ok(_credits(key_spent=...)), "not_reported"),
        (_ok(_credits(key_spent=None)), "not_reported"),
        (_ok(_credits(key_spent=float("nan"))), "invalid_response"),
        (_ok(_credits(key_spent=float("inf"))), "invalid_response"),
        (_ok(_credits(key_spent=-1)), "invalid_response"),
        (_ok(_credits(key_spent="0.001405")), "invalid_response"),
        (_ok(_credits(key_spent_since="2026-11-01T00:00:00Z")), "invalid_response"),
        (_ok(_credits(unit="eur")), "invalid_response"),
        (_error(401), "config:auth"),
        (_error(402), "config:billing"),
        (_error(429), "rate_limited"),
        (_error(503), "server_error"),
        (httpx.ConnectError(KEY), "transport"),
        (httpx.ReadTimeout(KEY), "timeout"),
        (RuntimeError(KEY), "exception"),
    ],
    ids=[
        "absent",
        "null",
        "nan",
        "inf",
        "negative",
        "string",
        "since_string",
        "not_usd",
        "401",
        "402",
        "429",
        "503",
        "transport",
        "timeout",
        "unforeseen",
    ],
)
def test_a_refuseless_read_with_no_figure_is_a_warning_naming_its_slug(caplog, response, reason):
    lines = _log(caplog, response=response)
    assert lines == [
        (
            "WARNING",
            (
                f"key spend provider=refuseless outcome=error reason={reason} "
                f"timeline={TIMELINE} delivery={DELIVERY}"
            ),
        )
    ]
    text = caplog.text
    assert KEY not in text and PROSE not in text and "refuseless.com" not in text


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        (_ok(_key(usage=...)), "not_reported"),
        (_ok(_key(usage=None)), "not_reported"),
        (_ok(_key(usage=float("nan"))), "invalid_response"),
        (_ok(_key(usage=-0.5)), "invalid_response"),
        (_ok(_key(usage=True)), "invalid_response"),
        (_ok(_key(usage=10**400)), "invalid_response"),
        (_ok({"data": []}), "invalid_response"),
        (_error(401), "config:auth"),
        (_error(429), "rate_limited"),
        (_error(500), "server_error"),
        (httpx.ConnectError(KEY), "transport"),
    ],
    ids=[
        "absent",
        "null",
        "nan",
        "negative",
        "bool",
        "overflow",
        "no_data",
        "401",
        "429",
        "500",
        "transport",
    ],
)
def test_an_openrouter_read_with_no_figure_is_a_warning_naming_its_slug(
    monkeypatch, caplog, response, reason
):
    monkeypatch.setenv("AI_PROVIDER", "openrouter")
    ((level, line),) = _log(caplog, "openrouter", OPENROUTER, response)
    assert level == "WARNING"
    assert line.startswith(f"key spend provider=openrouter outcome=error reason={reason} ")
    assert KEY not in caplog.text and PROSE not in caplog.text


def test_an_unusable_base_url_is_a_config_fault(monkeypatch, caplog):
    monkeypatch.setenv("AI_PROVIDER", "openrouter")
    monkeypatch.setenv("AI_BASE_URL", "openrouter.ai/api/v1")
    with (
        respx.mock(assert_all_mocked=True) as mock,
        caplog.at_level(logging.INFO, logger="basecradle_harness"),
    ):
        log_key_spend("openrouter", timeline=TIMELINE, delivery=None)
    assert not mock.calls
    assert "reason=config:base_url" in _lines(caplog)[0][1]


@pytest.mark.parametrize(
    ("vendor", "url"), [("refuseless", REFUSELESS), ("openrouter", OPENROUTER)]
)
def test_the_read_is_held_to_a_deadline_on_the_whole_attempt(monkeypatch, caplog, vendor, url):
    """A vendor that answers past the deadline is a timeout, however it kept the line alive."""
    monkeypatch.setenv("AI_PROVIDER", vendor)
    now = [1000.0]
    monkeypatch.setattr(_deadline, "_clock", lambda: now[0])

    def slow(request):
        now[0] += 11.0  # past the ten-second deadline before the answer arrives
        return httpx.Response(200, json=_credits() if vendor == "refuseless" else _key())

    ((_, line),) = _log(caplog, vendor, url, slow)
    assert "outcome=error reason=timeout" in line


# --- the wake writes it once, last before the bookend -------------------------------------------


@pytest.fixture
def platform():
    """The respx-mocked platform — its own copy, as `test_stop` keeps one."""
    with respx.mock(base_url=BC_URL, assert_all_called=False) as router:
        router.get("/users/dashboard").mock(return_value=httpx.Response(200, json=dashboard()))
        router.get(f"/timelines/{TIMELINE_UUID}").mock(
            return_value=httpx.Response(200, json=timeline())
        )
        router.post(f"/timelines/{TIMELINE_UUID}/messages").mock(
            return_value=httpx.Response(
                201, json={"message": message(uuid=REPLY, body="reply", mine=True)}
            )
        )
        router.get("/assets").mock(return_value=httpx.Response(200, json=asset_page()))
        router.get("/webhook_events").mock(return_value=httpx.Response(200, json=event_page()))
        router.get("/tasks").mock(return_value=httpx.Response(200, json=task_page()))
        yield router


class _Brain:
    """A brain that reports a vendor and answers once with text."""

    model = "brain-model"

    def __init__(self, provider: str, reply=None) -> None:
        self.provider = provider
        self.reply = reply

    def chat(self, messages, tools=None):
        from basecradle_harness import Message

        if isinstance(self.reply, BaseException):
            raise self.reply
        return Message.assistant("Thinking it over quietly.")


def test_a_refuseless_wake_writes_one_line_just_before_wake_end(platform, tmp_path, caplog):
    caplog.set_level("INFO", logger="basecradle_harness")
    serve_messages(platform, page(message(uuid=M0, body="How are the owls?")))
    route = platform.get(REFUSELESS).mock(return_value=httpx.Response(200, json=_credits()))
    agent, _ = build_wake(tmp_path, _Brain("refuseless"))
    agent.wake()
    heads = [plain(r.getMessage()) for r in caplog.records]
    spend = [i for i, m in enumerate(heads) if m.startswith("key spend ")]
    end = [i for i, m in enumerate(heads) if m.startswith("wake end ")]
    assert len(spend) == 1 and len(end) == 1 and spend[0] == end[0] - 1
    assert "outcome=ok spent=0.001405" in heads[spend[0]]
    assert route.call_count == 1


def test_a_stopped_wake_writes_no_line(platform, tmp_path, caplog):
    import signal

    caplog.set_level("INFO", logger="basecradle_harness")
    serve_messages(platform, page(message(uuid=M0, body="How are the owls?")))
    route = platform.get(REFUSELESS).mock(return_value=httpx.Response(200, json=_credits()))
    agent, _ = build_wake(tmp_path, _Brain("refuseless", WakeStopped(signal.SIGTERM)))
    with pytest.raises(WakeStopped):
        agent.wake()
    assert not route.called
    assert _lines(caplog) == []
    assert any("outcome=stopped" in plain(r.getMessage()) for r in caplog.records)
