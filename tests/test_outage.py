"""A wake that gives up on an outage exits 75 and tells the timeline once (issue #951).

`@origin`'s ruling of 2026-11-20 is the contract these tests pin: *when a wake gives up for a
temporary reason, the timeline is told once, plainly; and success resets it.* Before it, a rate
limit, a vendor that was down or a dropped connection that outlasted every in-call retry ended the
wake silently: the work stayed pending, nothing woke the agent again, and the person who wrote saw
nothing (2026-11-02).

The router's half is basecradle-router#527: a wake that exits ``75`` is tried four times in all
(at once, then at about 1, 5 and 30 minutes), and each attempt is told which it is through
``BASECRADLE_WAKE_ATTEMPT`` and ``BASECRADLE_WAKE_ATTEMPTS``. Only the last one speaks.
"""

from __future__ import annotations

import argparse
import logging

import httpx
import pytest
import respx

from basecradle_harness import (
    Message,
    MessagesTool,
    ProviderAuthError,
    ProviderBillingError,
    ProviderConnectionError,
    ProviderContextLengthError,
    ProviderPayloadTooLargeError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderServerError,
    ProviderTimeoutError,
    ToolCall,
)
from basecradle_harness import _wake as wake_module
from basecradle_harness._cleanup import enumerate_artifacts
from basecradle_harness._exceptions import TRANSIENT_FAILURES, ProviderAPIError
from basecradle_harness._observability import RED, RESET
from basecradle_harness._report import (
    STALL_DETAIL_CAP,
    OutageState,
    is_outage,
    outage_body,
)
from basecradle_harness._stop import WakeStopped
from basecradle_harness._wake import (
    EX_TEMPFAIL,
    WAKE_ATTEMPT_ENV,
    WAKE_ATTEMPTS_ENV,
    ClaimStore,
    MarkStore,
    last_attempt,
)
from tests.conftest import plain
from tests.test_wake import (
    BC_URL,
    M0,
    M4,
    PNG_BYTES,
    REPLY,
    TIMELINE_UUID,
    CountingProvider,
    RaisingProvider,
    _locked_problem,
    _posts,
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

MESSAGES_URL = f"/timelines/{TIMELINE_UUID}/messages"
DOWN = "Upstream provider returned 503: service temporarily unavailable"
NOTICE = "Automatic notice from this agent's harness"


@pytest.fixture
def platform():
    """The respx-mocked platform, with the timeline's post route named so a test can refuse it."""
    with respx.mock(base_url=BC_URL, assert_all_called=False) as router:
        router.get("/users/dashboard").mock(return_value=httpx.Response(200, json=dashboard()))
        router.get(f"/timelines/{TIMELINE_UUID}").mock(
            return_value=httpx.Response(200, json=timeline())
        )
        router.post(MESSAGES_URL, name="post").mock(
            return_value=httpx.Response(
                201, json={"message": message(uuid=REPLY, body="reply", mine=True)}
            )
        )
        router.get("/assets").mock(return_value=httpx.Response(200, json=asset_page()))
        router.get("/webhook_events").mock(return_value=httpx.Response(200, json=event_page()))
        router.get("/tasks").mock(return_value=httpx.Response(200, json=task_page()))
        router.get(path__regex=r"^/blobs/").mock(
            return_value=httpx.Response(200, content=PNG_BYTES)
        )
        yield router


@pytest.fixture
def attempt(monkeypatch):
    """Set the router's attempt pair, as `basecradle-router#527` exports it on a real wake."""

    def set_attempt(number: int, of: int = 4) -> None:
        monkeypatch.setenv(WAKE_ATTEMPT_ENV, str(number))
        monkeypatch.setenv(WAKE_ATTEMPTS_ENV, str(of))

    monkeypatch.delenv(WAKE_ATTEMPT_ENV, raising=False)
    monkeypatch.delenv(WAKE_ATTEMPTS_ENV, raising=False)
    return set_attempt


def _down() -> ProviderServerError:
    return ProviderServerError(DOWN, status_code=503)


def _agent(home, provider):
    """One router-spawned wake over `home`, with the engine's backoff neutered."""
    agent, _ = build_wake(home, provider, tools=[MessagesTool()])
    agent.harness.engine._sleep = lambda _seconds: None
    return agent


def _notices(platform) -> list[str]:
    return [body for body in _posts(platform) if body.startswith(NOTICE)]


def _wake_once(monkeypatch, agent) -> int:
    """The entrypoint's own verdict for `agent`'s wake — its exit code, through `_wake_once`."""
    monkeypatch.setattr(wake_module.WakeAgent, "from_env", classmethod(lambda cls, **_: agent))
    args = argparse.Namespace(timeline=TIMELINE_UUID, message=None, event=None, asset=None)
    stopped, code = wake_module._wake_once(args)
    assert stopped is None
    return code


# --- what an outage is ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [
        ProviderServerError("down", status_code=503),
        ProviderRateLimitError("slow down", status_code=429),
        ProviderConnectionError("connection reset"),
        ProviderTimeoutError("did not answer in time"),
        ProviderResponseError("unparseable body"),
    ],
    ids=["server_error", "rate_limited", "transport", "timeout", "invalid_response"],
)
def test_every_fault_the_engine_retries_is_an_outage(error):
    assert is_outage(error)


@pytest.mark.parametrize(
    "error",
    [
        ProviderBillingError("billing", status_code=402),
        ProviderAuthError("bad key", status_code=401),
        ProviderPayloadTooLargeError("too large", status_code=413),
        ProviderContextLengthError("too long", status_code=400),
        ProviderAPIError("malformed request", status_code=400),
        ValueError("a harness fault"),
    ],
    ids=["billing", "auth", "payload_too_large", "context_length", "api_error", "harness"],
)
def test_nothing_that_needs_a_human_or_a_fix_is_an_outage(error):
    """A config-class refusal is dead until a human acts, and a permanent one is a verdict on the
    content: a later wake would fail identically, so neither earns the router's patient retry."""
    assert not is_outage(error)


def test_the_outage_class_is_the_class_the_engine_retries():
    """One tuple at both distances: retried in the call, re-woken when it outlasts the retry."""
    from basecradle_harness._engine import _TRANSIENT

    assert _TRANSIENT is TRANSIENT_FAILURES


# --- the notice ----------------------------------------------------------------------------------


def test_the_notice_is_the_harness_speaking_for_itself_with_the_vendors_words():
    body = outage_body(provider="openrouter", exc=_down())

    assert body.startswith(f"{NOTICE} — its model did not write this. ")
    assert "its calls to OpenRouter kept failing" in body  # the vendor's own casing
    assert DOWN in body  # verbatim, never softened
    assert "Nothing is lost" in body
    assert "will be picked up on the next wake" in body
    assert "a new message on this timeline starts one" in body  # what makes the promise true


def test_the_notice_never_posts_a_credential_the_vendor_echoed():
    sentinel = "sk-or-v1-0f3a9c2b7d4e6f8a1b3c5d7e9f0a2b4c6d8e0f1a3b5c7d9e"
    echoed = ProviderServerError(
        "down",
        status_code=502,
        body='{"error": {"message": "Bad gateway for key Bearer ' + sentinel + '"}}',
    )

    body = outage_body(provider="openrouter", exc=echoed)

    assert sentinel not in body
    assert "Bad gateway for key [redacted]" in body


def test_the_notice_cuts_an_error_page_short():
    """A vendor that is down answers with an HTML page as often as with a sentence."""
    page_text = "<html>" + "x" * (STALL_DETAIL_CAP * 4) + "</html>"
    body = outage_body(provider="openai", exc=ProviderServerError(page_text, status_code=502))

    assert "x" * (STALL_DETAIL_CAP + 1) not in body
    assert "Nothing is lost" in body  # the sentence that says what to do survives the cut


def test_an_unknown_provider_still_reads_as_a_sentence():
    body = outage_body(provider=None, exc=_down())

    assert "its calls to the model provider kept failing" in body


# --- the last attempt ----------------------------------------------------------------------------


def test_no_attempt_variables_is_not_a_last_attempt(attempt):
    assert not last_attempt()


@pytest.mark.parametrize(("number", "of", "last"), [(1, 4, False), (3, 4, False), (4, 4, True)])
def test_the_last_attempt_is_the_one_whose_number_is_the_count(attempt, number, of, last):
    attempt(number, of)

    assert last_attempt() is last


def test_an_attempt_past_the_count_is_read_as_last(attempt):
    attempt(5, 4)

    assert last_attempt()


@pytest.mark.parametrize(
    ("number", "of"), [("four", "4"), ("0", "4"), ("4", ""), ("", "4"), ("-1", "-1")]
)
def test_a_broken_attempt_pair_is_not_last_and_says_so(monkeypatch, caplog, number, of):
    """The router's contract broken costs a notice not posted, never one posted early."""
    monkeypatch.setenv(WAKE_ATTEMPT_ENV, number)
    monkeypatch.setenv(WAKE_ATTEMPTS_ENV, of)

    with caplog.at_level(logging.WARNING, logger="basecradle_harness"):
        assert not last_attempt()

    assert any(WAKE_ATTEMPT_ENV in r.getMessage() for r in caplog.records)


# --- a wake that gives up on an outage -----------------------------------------------------------


def test_an_outage_before_the_last_attempt_posts_nothing(platform, tmp_path, attempt):
    """Every earlier attempt is followed by another, which may simply succeed."""
    for number in (1, 2, 3):
        attempt(number)
        serve_messages(platform, page(message(uuid=M0, body="are you there?")))
        agent = _agent(tmp_path, RaisingProvider(_down()))
        with pytest.raises(ProviderServerError):
            agent.wake()
        assert agent.gave_up is not None

    assert _posts(platform) == []
    assert MarkStore(tmp_path).get(TIMELINE_UUID) is None  # still pending


def test_a_wake_with_no_attempt_variables_posts_nothing(platform, tmp_path, attempt):
    """A wake run by hand, or by a router that predates the variables, behaves as before."""
    serve_messages(platform, page(message(uuid=M0, body="are you there?")))
    agent = _agent(tmp_path, RaisingProvider(_down()))

    with pytest.raises(ProviderServerError):
        agent.wake()

    assert _posts(platform) == []


def test_the_last_attempt_tells_the_timeline_once(platform, tmp_path, attempt, caplog):
    attempt(4)
    serve_messages(platform, page(message(uuid=M0, body="are you there?")))
    provider = RaisingProvider(_down())
    agent = _agent(tmp_path, provider)

    # The fault still ends the wake, as before.
    with (
        caplog.at_level(logging.INFO, logger="basecradle_harness"),
        pytest.raises(ProviderServerError),
    ):
        agent.wake()

    (notice,) = _notices(platform)
    assert DOWN in notice
    assert provider.prompts == []  # model-free: the model only raised
    assert MarkStore(tmp_path).get(TIMELINE_UUID) is None  # the message is kept, pending
    claim = ClaimStore(tmp_path).read(TIMELINE_UUID, M0, kind="messages")
    assert claim is not None and claim.phase == "in-flight"
    # One line under the telemetry contract, on the existing head, in the Failure tier.
    lines = [r.getMessage() for r in caplog.records]
    (line,) = [m for m in lines if m.startswith(f"{RED}wake reported_failure{RESET} ")]
    assert "kind=outage" in line
    assert "reason=server_error" in line
    assert "provider=openai" in line
    assert f"timeline={TIMELINE_UUID}" in line
    # The notice counts as what the wake put on the timeline.
    (end,) = [m for m in lines if "wake end" in m]
    assert "posted=1" in end and "outcome=error" in end


def test_one_outage_is_told_once_across_deliveries(platform, tmp_path, attempt):
    """A later delivery's last attempt during the same outage stays quiet."""
    attempt(4)
    for _ in range(3):
        serve_messages(platform, page(message(uuid=M0, body="are you there?")))
        with pytest.raises(ProviderServerError):
            _agent(tmp_path, RaisingProvider(_down())).wake()

    assert len(_notices(platform)) == 1


def test_success_resets_the_debounce_so_the_next_outage_is_told(platform, tmp_path, attempt):
    attempt(4)
    serve_messages(platform, page(message(uuid=M0, body="are you there?")))
    with pytest.raises(ProviderServerError):
        _agent(tmp_path, RaisingProvider(_down())).wake()
    assert OutageState(tmp_path).noted(TIMELINE_UUID)

    # The vendor answers again: the pending message is answered and the marker cleared.
    serve_messages(platform, page(message(uuid=M0, body="are you there?")))
    healthy = CountingProvider()
    _agent(tmp_path, healthy).wake()
    assert healthy.prompts != []
    assert not OutageState(tmp_path).noted(TIMELINE_UUID)

    # The next outage is a new one, and the timeline is told again.
    serve_messages(
        platform, page(message(uuid=M4, body="and now?"), message(uuid=M0, body="are you there?"))
    )
    with pytest.raises(ProviderServerError):
        _agent(tmp_path, RaisingProvider(_down())).wake()
    assert len(_notices(platform)) == 2


def test_a_refused_notice_is_tried_again_by_the_next_last_attempt(platform, tmp_path, attempt):
    """A post the platform refused told nobody, so the marker is not written."""
    attempt(4)
    platform.routes["post"].mock(return_value=httpx.Response(403, json=_locked_problem()))
    serve_messages(platform, page(message(uuid=M0, body="are you there?")))
    with pytest.raises(ProviderServerError):
        _agent(tmp_path, RaisingProvider(_down())).wake()
    assert not OutageState(tmp_path).noted(TIMELINE_UUID)

    platform.routes["post"].mock(
        return_value=httpx.Response(
            201, json={"message": message(uuid=REPLY, body="reply", mine=True)}
        )
    )
    serve_messages(platform, page(message(uuid=M0, body="are you there?")))
    with pytest.raises(ProviderServerError):
        _agent(tmp_path, RaisingProvider(_down())).wake()

    assert OutageState(tmp_path).noted(TIMELINE_UUID)
    assert len(_notices(platform)) == 2  # the refused attempt, then the one that landed


# --- the exit code -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [
        ProviderServerError(DOWN, status_code=503),
        ProviderRateLimitError("rate limited", status_code=429),
        ProviderConnectionError("connection reset by peer"),
        ProviderTimeoutError("did not answer in time"),
    ],
    ids=["server_error", "rate_limited", "transport", "timeout"],
)
def test_a_wake_that_gives_up_on_an_outage_exits_75(platform, tmp_path, monkeypatch, error):
    serve_messages(platform, page(message(uuid=M0, body="are you there?")))

    assert _wake_once(monkeypatch, _agent(tmp_path, RaisingProvider(error))) == EX_TEMPFAIL == 75


def test_an_auth_failure_keeps_exit_1(platform, tmp_path, monkeypatch):
    """Dead until a human fixes the key: the router's patient schedule would only repeat it."""
    serve_messages(platform, page(message(uuid=M0, body="are you there?")))
    error = ProviderAuthError("invalid api key", status_code=401)

    assert _wake_once(monkeypatch, _agent(tmp_path, RaisingProvider(error))) == 1


def test_a_reported_failure_keeps_exit_0(platform, tmp_path, monkeypatch):
    """Out of funds is reported and handled by the wake itself, exactly as before."""
    serve_messages(platform, page(message(uuid=M0, body="are you there?")))
    error = ProviderBillingError("billing", status_code=402)

    assert _wake_once(monkeypatch, _agent(tmp_path, RaisingProvider(error))) == 0


def test_a_healthy_wake_keeps_exit_0(platform, tmp_path, monkeypatch):
    serve_messages(platform, page(message(uuid=M0, body="are you there?")))

    assert _wake_once(monkeypatch, _agent(tmp_path, CountingProvider())) == 0


# --- a resume that fails on an outage ------------------------------------------------------------


class _SpeaksThenFails:
    """A turn that posts once and then dies on `error`; a later wake's resume dies on it too."""

    provider, model = "openrouter", "z-ai/glm-5.2"

    def __init__(self, error: Exception, *, speaks: bool = True) -> None:
        self.error = error
        self.speaks = speaks
        self.calls = 0

    def chat(self, messages, tools=None):
        self.calls += 1
        if self.speaks and self.calls == 1:
            return Message.assistant(
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="messages",
                        arguments={"action": "create", "body": "On it."},
                    )
                ]
            )
        raise self.error


def _died_mid_tool_chain(platform, home, error):
    serve_messages(platform, page(message(uuid=M0, body="audit the repo")))
    with pytest.raises(type(error)):
        _agent(home, _SpeaksThenFails(error)).wake()


def test_a_resume_that_fails_on_an_outage_exits_75_and_tells_the_last_attempt(
    platform, tmp_path, monkeypatch, attempt
):
    """The resume's failure is absorbed so the wake can answer the rest, but its item still waits
    on a wake nothing else will start, so this wake gave up on it."""
    _died_mid_tool_chain(platform, tmp_path, _down())

    attempt(4)
    serve_messages(platform, page(message(uuid=M0, body="audit the repo")))
    brain = _SpeaksThenFails(_down(), speaks=False)
    agent = _agent(tmp_path, brain)

    assert _wake_once(monkeypatch, agent) == EX_TEMPFAIL
    assert brain.calls >= 1  # it did resume
    assert len(_notices(platform)) == 1
    claim = ClaimStore(tmp_path).read(TIMELINE_UUID, M0, kind="messages")
    assert claim is not None and claim.phase == "in-flight"  # kept, for the next wake


def test_a_resume_that_timed_out_on_the_turn_is_not_an_outage(
    platform, tmp_path, monkeypatch, attempt
):
    """Issue #589: a resume out of the harness's own budget is the turn's fault and counts toward
    the ceiling. A patient router retry would spend that ceiling on a slow vendor within minutes,
    so it keeps the exit and the silence it had."""
    timed_out = ProviderTimeoutError("did not answer in time")
    _died_mid_tool_chain(platform, tmp_path, timed_out)

    attempt(4)
    serve_messages(platform, page(message(uuid=M0, body="audit the repo")))
    brain = _SpeaksThenFails(timed_out, speaks=False)
    agent = _agent(tmp_path, brain)

    assert _wake_once(monkeypatch, agent) == 0
    assert brain.calls >= 1
    assert _notices(platform) == []
    claim = ClaimStore(tmp_path).read(TIMELINE_UUID, M0, kind="messages")
    assert claim is not None and (claim.phase, claim.resumes) == ("in-flight", 1)


def test_a_resume_that_fails_on_a_fault_that_is_not_an_outage_keeps_exit_0(
    platform, tmp_path, monkeypatch, attempt
):
    """The same ``except`` absorbs faults that are not an outage, and they keep the code they had."""
    _died_mid_tool_chain(platform, tmp_path, _down())

    attempt(4)
    serve_messages(platform, page(message(uuid=M0, body="audit the repo")))
    brain = _SpeaksThenFails(ProviderAuthError("invalid api key", status_code=401), speaks=False)
    agent = _agent(tmp_path, brain)

    assert _wake_once(monkeypatch, agent) == 0
    assert brain.calls >= 1  # it did resume, and failed there
    assert _notices(platform) == []
    claim = ClaimStore(tmp_path).read(TIMELINE_UUID, M0, kind="messages")
    assert claim is not None and claim.phase == "in-flight"


def test_the_latest_failure_of_an_item_decides(platform, tmp_path):
    """An item a later pass failed some other way is no longer waiting on the outage."""
    serve_messages(platform, page())
    agent = _agent(tmp_path, CountingProvider())

    agent._note_outage("messages", M0, _down())
    assert agent._outage_left() is not None
    agent._note_outage("messages", M0, ProviderAuthError("invalid api key", status_code=401))

    assert agent._outage_left() is None


def test_an_item_a_later_pass_finished_is_not_given_up_on(platform, tmp_path):
    serve_messages(platform, page())
    agent = _agent(tmp_path, CountingProvider())
    agent._note_outage("messages", M0, _down())
    assert agent.claims.claim(TIMELINE_UUID, M0, kind="messages")
    assert agent._outage_left() is not None  # in flight: still waiting

    agent.claims.commit(TIMELINE_UUID, M0, kind="messages")

    assert agent._outage_left() is None


# --- success is a call, and a stop is a stop -----------------------------------------------------


def test_the_first_successful_call_resets_the_debounce_even_if_the_turn_then_fails(
    platform, tmp_path, attempt
):
    """A turn whose first call answers and whose next one fails is a new outage, and is told."""
    OutageState(tmp_path).note(TIMELINE_UUID)  # an earlier outage, already announced
    attempt(4)
    serve_messages(platform, page(message(uuid=M0, body="audit the repo")))

    with pytest.raises(ProviderServerError):
        _agent(tmp_path, _SpeaksThenFails(_down())).wake()

    assert _posts(platform)[0] == "On it."  # the call that answered
    assert len(_notices(platform)) == 1  # the new outage was told
    assert OutageState(tmp_path).noted(TIMELINE_UUID)


def test_a_stop_while_the_notice_posts_is_settled_as_a_stop(platform, tmp_path, attempt, caplog):
    """The stop lands inside the outage handler, where the stop handler below cannot catch it."""
    attempt(4)
    serve_messages(platform, page(message(uuid=M0, body="are you there?")))
    agent = _agent(tmp_path, RaisingProvider(_down()))

    def stopped(*_args, **_kwargs):
        raise WakeStopped()

    agent._post = stopped

    with caplog.at_level(logging.INFO, logger="basecradle_harness"), pytest.raises(WakeStopped):
        agent.wake()

    claim = ClaimStore(tmp_path).read(TIMELINE_UUID, M0, kind="messages")
    assert claim is not None and claim.phase == "abandoned"  # settled, never resumed as a crash
    (end,) = [r.getMessage() for r in caplog.records if "wake end" in r.getMessage()]
    assert "outcome=stopped" in plain(end) and "signal=TERM" in plain(end)


# --- what the notice leaves behind ---------------------------------------------------------------


def test_the_orphan_sweep_purges_a_deleted_timelines_outage_marker(tmp_path):
    """Whatever creates, cleans up: the marker goes with the timeline it was written for."""
    OutageState(tmp_path).note(TIMELINE_UUID)

    artifacts = enumerate_artifacts(tmp_path)

    assert artifacts[TIMELINE_UUID] == [tmp_path / "outage" / f"{TIMELINE_UUID}.notified"]
