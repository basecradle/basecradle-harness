"""A turn the recovery cannot finish stalls, visibly — it never loops (issue #589).

On 2026-09-29 @glm-5.2's wake died mid-tool-chain on a model call that needed more time than it
was given, and the recovery did what it was built to do: it resumed the turn — with the same
accumulated context, into the same failure — and died the same way. Then it did it again, and
again. Four resumes of one item, and the loop broke only because the peer changed the ask.

The founder's ruling is the contract these tests pin: *a bad ask should cost one wasted wake and a
visible stall, never a crash loop.* After `RESUME_CEILING` failed resumes the harness stops, posts
a stall note to the timeline in its own words, logs a WARNING, and marks the item — so nothing
stalls behind it and nobody is left guessing.
"""

from __future__ import annotations

import json
import logging

import httpx
import pytest
import respx

from basecradle_harness import (
    Message,
    MessagesTool,
    ProviderBillingError,
    ProviderConnectionError,
    ProviderServerError,
    ProviderTimeoutError,
    ToolCall,
    WakeBreaker,
)
from basecradle_harness._idempotency import STALL_NOTE, key
from basecradle_harness._report import (
    STALL_DETAIL_CAP,
    STALL_NOTHING_WRITTEN,
    STALL_PAST_BUDGET,
    stall_body,
    stall_detail,
)
from basecradle_harness._wake import (
    RESUME_CEILING,
    Claim,
    ClaimStore,
    MarkStore,
    _payload,
    _read_claim,
)
from tests.test_truncation import _CutOff, _CutOffThenDown
from tests.test_wake import (
    BC_URL,
    M0,
    M1,
    M2,
    PNG_BYTES,
    REPLY,
    TIMELINE_UUID,
    CountingProvider,
    ScriptedMessages,
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

BODY = "Clone the repo, run the audit, fix every finding, rerun it, and write up what changed."
TIMED_OUT = "OpenRouter did not answer in time: The read operation timed out"
MESSAGES_URL = f"/timelines/{TIMELINE_UUID}/messages"


@pytest.fixture
def platform():
    """The respx-mocked platform — its own copy, for the reason `test_resume` keeps one."""
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


class _WorksThenStalls:
    """The incident's brain: it gets partway — a tool call that posts — and then never answers.

    Every call after the first raises the timeout the engine gives up on (it has already retried
    it once, at twice the budget). ``speaks=False`` is the same brain on a later wake, which never
    gets as far as a tool call because the turn it is resuming is the one that is too heavy.
    """

    provider, model = "openrouter", "z-ai/glm-5.2"

    def __init__(self, *, speaks: bool = True) -> None:
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
                        arguments={"action": "create", "body": "On it — cloning now."},
                    )
                ]
            )
        raise ProviderTimeoutError(TIMED_OUT)


def _wake(home, brain):
    """One router-spawned wake over `home`, with the engine's backoff neutered."""
    agent, _ = build_wake(home, brain, tools=[MessagesTool()])
    agent.harness.engine._sleep = lambda _seconds: None
    return agent


def _claim(home, uuid=M0) -> Claim:
    claim = ClaimStore(home).read(TIMELINE_UUID, uuid, kind="messages")
    assert claim is not None
    return claim


def _died_mid_tool_chain(platform, home, *uuids):
    """Wake 1: the turn posts once, then its next model call never answers and the wake dies."""
    serve_messages(platform, page(*(message(uuid=u, body=BODY) for u in uuids)))
    with pytest.raises(ProviderTimeoutError):
        _wake(home, _WorksThenStalls()).wake()


def _stall_posts(platform) -> list[httpx.Request]:
    return [
        call.request
        for call in platform.calls
        if call.request.method == "POST"
        and call.request.url.path.endswith("/messages")
        and "Automatic notice" in json.loads(call.request.content)["message"]["body"]
    ]


# --- the incident, end to end -------------------------------------------------------------------


def test_a_turn_that_cannot_be_finished_stalls_instead_of_looping(platform, tmp_path, caplog):
    """The wake that died, then every resume the ceiling allows, then a stall — and then nothing."""
    _died_mid_tool_chain(platform, tmp_path, M0)
    assert _posts(platform) == ["On it — cloning now."]

    for resume in range(1, RESUME_CEILING):
        serve_messages(platform, page(message(uuid=M0, body=BODY)))
        brain = _WorksThenStalls(speaks=False)
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="basecradle_harness"):
            _wake(tmp_path, brain).wake()  # the failed resume never takes the wake down
        assert brain.calls >= 1, "it did resume"
        claim = _claim(tmp_path)
        assert (claim.phase, claim.resumes) == ("in-flight", resume)
        assert not _stall_posts(platform)
        # Left waiting on a wake nothing is scheduled to start, and the journal says so (#596).
        assert any(
            "wake deferred" in r.getMessage() and "reason=resume_failed" in r.getMessage()
            for r in caplog.records
        )

    serve_messages(platform, page(message(uuid=M0, body=BODY)))
    with caplog.at_level(logging.WARNING, logger="basecradle_harness"):
        _wake(tmp_path, _WorksThenStalls(speaks=False)).wake()

    # The stall note: posted once, in the harness's own words, keyed so a repeat cannot double it.
    (note,) = _stall_posts(platform)
    body = json.loads(note.content)["message"]["body"]
    assert body.startswith("Automatic notice from this agent's harness — its model did not write")
    assert f"{RESUME_CEILING} attempts to resume it failed" in body
    assert "your message" in body
    assert TIMED_OUT in body  # the vendor's own words, verbatim
    assert "1 tool call toward it" in body  # what it was doing
    assert "What would help" in body
    assert note.headers["Idempotency-Key"] == key(  # anchored on the turn, like its other keys
        timeline=TIMELINE_UUID, anchor=M0, kind=STALL_NOTE, ordinal=1
    )

    # The item is marked: settled, so the record passes it and nothing stalls behind it.
    claim = _claim(tmp_path)
    assert claim.phase == "abandoned"
    assert "stalled after" in claim.reason
    assert MarkStore(tmp_path).get(TIMELINE_UUID) == M0

    # And the journal says so, at WARNING — the founder's level for it, not a page.
    stalled = [r for r in caplog.records if "wake stalled" in r.getMessage()]
    assert [r.levelname for r in stalled] == ["WARNING"]
    assert f"resumes={RESUME_CEILING}" in stalled[0].getMessage()

    # The next wake does nothing at all: no model call, no second note.
    serve_messages(platform, page(message(uuid=M0, body=BODY)))
    after = _WorksThenStalls(speaks=False)
    _wake(tmp_path, after).wake()
    assert after.calls == 0
    assert len(_stall_posts(platform)) == 1


class _Killed(BaseException):
    """What a `SIGKILL` looks like from inside the process: nothing any `except Exception` catches."""


class _DiesMidResume:
    """A resume the box kills: the model call never returns and never raises anything catchable."""

    provider, model = "openrouter", "z-ai/glm-5.2"

    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages, tools=None):
        self.calls += 1
        raise _Killed


def test_a_resume_the_box_killed_counts_and_the_next_wake_stalls_without_the_model(
    platform, tmp_path
):
    """The count is written when a resume *starts*, so a resume that never reports back — the
    process killed mid-call, the shape a crash loop is actually made of — still counts. After the
    ceiling's worth of them, the next wake stalls the item before calling the model at all."""
    _died_mid_tool_chain(platform, tmp_path, M0)
    for resume in range(1, RESUME_CEILING + 1):
        serve_messages(platform, page(message(uuid=M0, body=BODY)))
        with pytest.raises(_Killed):
            _wake(tmp_path, _DiesMidResume()).wake()
        assert _claim(tmp_path).resumes == resume  # counted though nothing reported back

    serve_messages(platform, page(message(uuid=M0, body=BODY)))
    brain = _WorksThenStalls(speaks=False)
    _wake(tmp_path, brain).wake()

    assert brain.calls == 0
    (note,) = _stall_posts(platform)
    body = json.loads(note.content)["message"]["body"]
    assert "the last of them never reported back" in body
    assert _claim(tmp_path).phase == "abandoned"


def test_a_wake_killed_while_the_breaker_holds_it_has_not_resumed_anything(platform, tmp_path):
    """The breaker admits a resume *before* the take-over, never after (issue #592). The take-over
    is what counts a resume against the ceiling, and a wake the box kills while it waits out a trip
    never reached the model — counted, a trip during a bad hour would stall a turn no resume had
    ever tried to finish. The next wake, finishing the trip, still has the whole ceiling."""

    def killed(_seconds):
        raise _Killed

    def resume_with(brain, *, sleep):
        breaker = WakeBreaker(tmp_path, max_wakes=1, sleep=sleep)
        agent, _ = build_wake(tmp_path, brain, tools=[MessagesTool()], breaker=breaker)
        agent.harness.engine._sleep = lambda _seconds: None
        serve_messages(platform, page(message(uuid=M0, body=BODY)))
        return agent

    _died_mid_tool_chain(platform, tmp_path, M0)
    standing = WakeBreaker(tmp_path, max_wakes=1)
    assert standing.admit(TIMELINE_UUID).tripped  # the dead wake was the first; this, the second

    brain = _DiesMidResume()
    with pytest.raises(_Killed):
        resume_with(brain, sleep=killed).wake()
    assert brain.calls == 0
    assert (_claim(tmp_path).phase, _claim(tmp_path).resumes) == ("in-flight", 0)

    with pytest.raises(_Killed):
        resume_with(_DiesMidResume(), sleep=lambda _seconds: None).wake()
    assert _claim(tmp_path).resumes == 1  # the first resume that reached the model
    assert not standing.tripped(TIMELINE_UUID)


def test_a_hold_in_a_resume_still_folds_in_what_landed_meanwhile(platform, tmp_path):
    """A resume that `_recover` runs ahead of the batch can be the wake's first model work, so the
    breaker can hold it there — after the message list was read (issue #592). A message that lands
    during that hold must still be answered by this wake, even when everything in the list was the
    turn being resumed and the batch would otherwise be empty."""
    _died_mid_tool_chain(platform, tmp_path, M0)
    assert WakeBreaker(tmp_path, max_wakes=1).admit(TIMELINE_UUID).tripped

    scripted = ScriptedMessages(platform, message(uuid=M0, body=BODY))

    def arrive(_seconds):  # the hold's sleep: a message lands while the resume waits
        scripted.arrive(message(uuid=M1, body="are you there?"))

    brain = CountingProvider()
    agent, _ = build_wake(
        tmp_path, brain, tools=[MessagesTool()], breaker=WakeBreaker(tmp_path, sleep=arrive)
    )
    agent.harness.engine._sleep = lambda _seconds: None

    agent.wake()

    assert _claim(tmp_path).phase == "done"  # the resumed turn finished…
    assert brain.prompts == ["[2026-06-04T00:00:00.000Z] john: are you there?"]  # …and M1 answered
    assert MarkStore(tmp_path).get(TIMELINE_UUID) == M1


def test_a_resume_hold_that_ends_in_a_latch_claims_nothing_more(platform, tmp_path):
    """The fold after a resume's hold is for a wake that will go on to answer. When the resume hits
    a wall that latches the wake (out of funds here), nothing more is answered this wake, so the
    message that landed mid-hold is left unclaimed for its own delivery — never claimed only to be
    handed to the next wake as an orphan."""
    _died_mid_tool_chain(platform, tmp_path, M0)
    assert WakeBreaker(tmp_path, max_wakes=1).admit(TIMELINE_UUID).tripped
    scripted = ScriptedMessages(platform, message(uuid=M0, body=BODY))

    def arrive(_seconds):
        scripted.arrive(message(uuid=M1, body="are you there?"))

    class _Unfunded:
        provider, model = "openrouter", "z-ai/glm-5.2"

        def chat(self, messages, tools=None):
            raise ProviderBillingError("Insufficient credits", status_code=402)

    agent, _ = build_wake(
        tmp_path, _Unfunded(), tools=[MessagesTool()], breaker=WakeBreaker(tmp_path, sleep=arrive)
    )
    agent.harness.engine._sleep = lambda _seconds: None
    agent.wake()

    assert ClaimStore(tmp_path).read(TIMELINE_UUID, M1, kind="messages") is None


@pytest.mark.parametrize(
    "fault",
    [
        ProviderBillingError("Insufficient credits", status_code=402),
        ProviderServerError("upstream error", status_code=502),
        ProviderConnectionError("Could not reach OpenRouter: connection reset"),
    ],
    ids=["out-of-funds", "5xx", "transport"],
)
def test_an_outage_during_a_resume_never_counts_toward_the_ceiling(platform, tmp_path, fault):
    """A fault of the provider says nothing about whether the turn can be finished — and it is the
    path an outage takes. Counted, a funding gap or a 5xx storm two wakes long would stall every
    unfinished turn on the box; uncounted, the item waits for the cause to clear, as #336 decided a
    peer's message must."""

    class _Outage:
        provider, model = "openrouter", "z-ai/glm-5.2"

        def chat(self, messages, tools=None):
            raise fault

    _died_mid_tool_chain(platform, tmp_path, M0)
    for _ in range(RESUME_CEILING + 2):
        serve_messages(platform, page(message(uuid=M0, body=BODY)))
        _wake(tmp_path, _Outage()).wake()
        claim = _claim(tmp_path)
        assert (claim.phase, claim.resumes) == ("in-flight", 0)

    assert not _stall_posts(platform)


def test_a_continuation_that_writes_nothing_is_not_progress(platform, tmp_path):
    """A model that spends its whole output budget before a word — a reasoning model thinking to
    the cap — is cut off with nothing written, every time. Clearing the count for that would loop
    for as long as the truncation notes took to fill the context; it counts, and it stalls — inside
    the wake that cut it off, since #596 finishes a turn where it was cut off."""
    serve_messages(platform, page(message(uuid=M0, body=BODY)))
    agent, _ = build_wake(tmp_path, _CutOff(Message.assistant(content="")), tools=[MessagesTool()])
    brain = agent.harness.provider

    agent.wake()

    assert brain.calls == 1 + RESUME_CEILING  # the fresh turn, then two continuations
    (note,) = _stall_posts(platform)
    assert (
        "ran out of its output budget before writing anything"
        in json.loads(note.content)["message"]["body"]
    )
    assert _claim(tmp_path).phase == "abandoned"


def test_a_stall_note_the_platform_refuses_is_never_a_silent_drop(platform, tmp_path, caplog):
    """Only a posted note licenses the abandon. Refused, the item stays in flight at the ceiling,
    and the next wake stalls it again — without a model call — once the timeline takes the post.
    Meanwhile it is waiting on a wake nothing is scheduled to start, and the journal says so
    (`wake deferred … reason=stall_note_refused`, issue #596)."""
    _died_mid_tool_chain(platform, tmp_path, M0)
    for _ in range(RESUME_CEILING - 1):
        serve_messages(platform, page(message(uuid=M0, body=BODY)))
        _wake(tmp_path, _WorksThenStalls(speaks=False)).wake()
    platform.routes["post"].mock(
        return_value=httpx.Response(423, json={"error": "This timeline is locked."})
    )

    serve_messages(platform, page(message(uuid=M0, body=BODY)))
    with caplog.at_level(logging.WARNING, logger="basecradle_harness"):
        _wake(tmp_path, _WorksThenStalls(speaks=False)).wake()  # the last resume, a refused note

    refused = [
        r.getMessage()
        for r in caplog.records
        if "wake deferred" in r.getMessage() and "reason=stall_note_refused" in r.getMessage()
    ]
    assert len(refused) == 1 and f"item={M0}" in refused[0]
    claim = _claim(tmp_path)
    assert (claim.phase, claim.resumes) == ("in-flight", RESUME_CEILING)
    assert TIMED_OUT in claim.reason  # what the refused note said, kept for the next one
    assert MarkStore(tmp_path).get(TIMELINE_UUID) is None  # nothing passed it

    platform.routes["post"].mock(
        return_value=httpx.Response(
            201, json={"message": message(uuid=REPLY, body="reply", mine=True)}
        )
    )
    serve_messages(platform, page(message(uuid=M0, body=BODY)))
    retried = _WorksThenStalls(speaks=False)
    _wake(tmp_path, retried).wake()

    assert retried.calls == 0
    assert _claim(tmp_path).phase == "abandoned"
    first, second = _stall_posts(platform)
    # Both carried the one key, so the platform keeps one note; and the second said what the first
    # would have — the detail rode the claim across the refusal.
    assert first.headers["Idempotency-Key"] == second.headers["Idempotency-Key"]
    assert json.loads(first.content) == json.loads(second.content)


def test_a_stalled_batch_is_stalled_whole_with_one_note(platform, tmp_path):
    """A turn answers a batch, so its stall is the batch's: one note naming them, every message in
    it marked — none committed as if answered, and none resumed again on its own."""
    MarkStore(tmp_path).set(TIMELINE_UUID, M0)  # a prior wake answered M0; M1 and M2 are new
    inbox = page(
        message(uuid=M2, body="and one more thing"),
        message(uuid=M1, body=BODY),
        message(uuid=M0, body="answered already"),
    )
    serve_messages(platform, inbox)
    with pytest.raises(ProviderTimeoutError):
        _wake(tmp_path, _WorksThenStalls()).wake()
    for _ in range(RESUME_CEILING):
        serve_messages(platform, inbox)
        _wake(tmp_path, _WorksThenStalls(speaks=False)).wake()

    (note,) = _stall_posts(platform)
    assert "your last 2 messages" in json.loads(note.content)["message"]["body"]
    assert _claim(tmp_path, M1).phase == "abandoned"
    assert _claim(tmp_path, M2).phase == "abandoned"
    assert _claim(tmp_path, M2).reason == "stalled with the turn that carried it"
    assert MarkStore(tmp_path).get(TIMELINE_UUID) == M2  # nothing is left behind the cursor
    assert note.headers["Idempotency-Key"] == key(
        timeline=TIMELINE_UUID, anchor=M1, kind=STALL_NOTE, ordinal=1
    )


def test_a_wake_that_dies_mid_stall_leaves_one_note_and_nothing_resumed(
    platform, tmp_path, monkeypatch
):
    """The batch is settled before the item that carries the count, so a wake dying between the two
    leaves that item — at the ceiling — for the next wake to stall, and never a batch-mate that
    looks like a fresh orphan. And the note is keyed on the turn, so the second stall's post is the
    same note, whichever message it went through."""
    MarkStore(tmp_path).set(TIMELINE_UUID, M0)
    inbox = page(message(uuid=M2, body="and one more thing"), message(uuid=M1, body=BODY))
    serve_messages(platform, inbox)
    with pytest.raises(ProviderTimeoutError):
        _wake(tmp_path, _WorksThenStalls()).wake()
    for _ in range(RESUME_CEILING - 1):
        serve_messages(platform, inbox)
        _wake(tmp_path, _WorksThenStalls(speaks=False)).wake()

    real_abandon = ClaimStore.abandon

    def dies_on_the_lead(self, timeline, uuid, *, kind, reason):
        if uuid == M1:
            raise _Killed
        real_abandon(self, timeline, uuid, kind=kind, reason=reason)

    monkeypatch.setattr(ClaimStore, "abandon", dies_on_the_lead)
    serve_messages(platform, inbox)
    with pytest.raises(_Killed):
        _wake(tmp_path, _WorksThenStalls(speaks=False)).wake()
    monkeypatch.setattr(ClaimStore, "abandon", real_abandon)
    assert _claim(tmp_path, M2).phase == "abandoned"
    assert _claim(tmp_path, M1).phase == "in-flight"

    serve_messages(platform, inbox)
    after = _WorksThenStalls(speaks=False)
    _wake(tmp_path, after).wake()

    assert after.calls == 0  # stalled from the record, never resumed again
    assert _claim(tmp_path, M1).phase == "abandoned"
    keys = {request.headers["Idempotency-Key"] for request in _stall_posts(platform)}
    assert len(keys) == 1


def test_a_continuation_cut_off_at_the_budget_is_progress_not_a_failure(platform, tmp_path):
    """Issue #490's resume is not what the ceiling counts *across wakes*. A long answer continued
    once per wake is converging — each continuation carries more of it — so the resume a wake makes
    of an earlier wake's turn clears the count, and it is never stalled for being long. (Inside one
    wake the capital ruled otherwise, #596; here each wake's own continuation is unreachable, so
    only the cross-wake resumes run.)"""
    serve_messages(platform, page(message(uuid=M0, body=BODY)))
    build_wake(tmp_path, _CutOffThenDown())[0].wake()

    for _ in range(RESUME_CEILING + 1):
        serve_messages(platform, page(message(uuid=M0, body=BODY)))
        brain = _CutOffThenDown(Message.assistant(content="…and more of it"))
        build_wake(tmp_path, brain)[0].wake()
        assert brain.calls == 2  # resumed (and progressed), then its in-wake continuation failed
        claim = _claim(tmp_path)
        assert (claim.phase, claim.resumes) == ("in-flight", 0)

    assert not _stall_posts(platform)


def test_a_continuation_the_box_kills_inside_the_wake_still_counts(platform, tmp_path):
    """The in-wake continuation is a resume like any other: its count is written when it *starts*,
    so a kill there counts toward the ceiling exactly as a killed cross-wake resume does (#589)."""

    class _CutOffThenKilled(_CutOff):
        def chat(self, messages, tools=None):
            if self.calls >= 1:
                self.calls += 1
                raise _Killed
            return super().chat(messages, tools)

    serve_messages(platform, page(message(uuid=M0, body=BODY)))
    with pytest.raises(_Killed):
        build_wake(tmp_path, _CutOffThenKilled())[0].wake()

    claim = _claim(tmp_path)
    assert (claim.phase, claim.resumes) == ("in-flight", 1)


# --- the count, on the claim --------------------------------------------------------------------


def test_the_count_rides_the_take_over_and_its_token(tmp_path):
    store = ClaimStore(tmp_path, wake="dead")
    store.claim(TIMELINE_UUID, M0, kind="messages")
    store._write(
        store._path(TIMELINE_UUID, "messages", M0),
        Claim(phase="in-flight", pid=1, wake="dead", at=0.0),
    )

    recoverer = ClaimStore(tmp_path, wake="recoverer")
    assert recoverer.reclaim(TIMELINE_UUID, M0, kind="messages", owner="dead", resumes=2)

    assert _claim(tmp_path).resumes == 2
    token = recoverer._token(recoverer._path(TIMELINE_UUID, "messages", M0), M0, "dead")
    assert _read_claim(token).resumes == 2  # a take-over that dies before the claim still counts


def test_a_heartbeat_mid_resume_keeps_the_count(tmp_path):
    """The heartbeat rewrites the claim every few minutes of a long resume; a fresh record there
    would reset the ceiling the crash loop is counted against."""
    store = ClaimStore(tmp_path, wake="dead")
    store.claim(TIMELINE_UUID, M0, kind="messages")
    store._write(
        store._path(TIMELINE_UUID, "messages", M0),
        Claim(phase="in-flight", pid=1, wake="dead", at=0.0),
    )
    recoverer = ClaimStore(tmp_path, wake="recoverer")
    recoverer.reclaim(TIMELINE_UUID, M0, kind="messages", owner="dead", resumes=1)
    recoverer._held = {key_: (0.0, 0.0) for key_ in recoverer._held}  # due now

    recoverer.beat()

    claim = _claim(tmp_path)
    assert claim.resumes == 1
    assert claim.at > 0.0  # it did refresh


def test_only_the_wake_holding_the_claim_changes_its_count(tmp_path):
    store = ClaimStore(tmp_path, wake="dead")
    store.claim(TIMELINE_UUID, M0, kind="messages")
    store._write(
        store._path(TIMELINE_UUID, "messages", M0),
        Claim(phase="in-flight", pid=1, wake="dead", at=0.0),
    )
    recoverer = ClaimStore(tmp_path, wake="recoverer")
    recoverer.reclaim(TIMELINE_UUID, M0, kind="messages", owner="dead", resumes=2)

    ClaimStore(tmp_path, wake="someone-else").recount(TIMELINE_UUID, M0, kind="messages", resumes=0)
    assert _claim(tmp_path).resumes == 2  # not theirs to change

    recoverer.recount(TIMELINE_UUID, M0, kind="messages", resumes=1, reason="last error: x")
    claim = _claim(tmp_path)
    assert (claim.resumes, claim.reason) == (1, "last error: x")


def test_a_record_without_a_count_reads_as_none_and_writes_as_before():
    """Every claim written before issue #589 reads as zero resumes, and a claim that never needed
    the field is written byte for byte as it was."""
    assert "resumes" not in _payload(Claim(phase="in-flight", pid=7, wake="w", at=1.0))
    assert _payload(Claim(phase="in-flight", resumes=2))["resumes"] == 2


@pytest.mark.parametrize("raw", ['"two"', "true", "-1", "null", "2.5"])
def test_an_unreadable_count_reads_as_none(tmp_path, raw):
    path = tmp_path / "claim"
    path.write_text(f'{{"phase": "in-flight", "pid": 1, "wake": "w", "at": 0, "resumes": {raw}}}')
    assert _read_claim(path).resumes == 0


# --- the note, in the harness's own words -------------------------------------------------------


def test_the_note_says_what_it_was_doing_that_it_could_not_finish_and_what_would_help():
    body = stall_body(
        item="your message",
        resumes=2,
        tool_calls=8,
        detail=stall_detail(ProviderTimeoutError(TIMED_OUT)),
    )
    assert "Automatic notice from this agent's harness — its model did not write this." in body
    assert "could not finish working on your message" in body
    assert "the turn stopped partway, and 2 attempts to resume it failed" in body
    assert "8 tool calls toward it" in body
    assert f"last error: {TIMED_OUT}" in body
    assert "What would help: send it again, ideally split into smaller steps." in body
    assert body.endswith("If this keeps happening, the model provider may be having trouble.")


@pytest.mark.parametrize("detail", [STALL_PAST_BUDGET, STALL_NOTHING_WRITTEN])
def test_a_budget_stall_names_the_budget_not_the_provider(detail):
    """A turn that kept running out of its output budget is not the provider having trouble, and a
    note that said so would send the reader looking in the wrong place (issue #596)."""
    body = stall_body(item="your message", resumes=2, tool_calls=0, detail=detail)
    assert body.endswith(
        "If this keeps happening, this agent's output budget may be too small for answers like "
        "this one."
    )
    assert "provider" not in body


def test_an_internal_fault_is_named_never_quoted():
    """A harness fault's text can carry a path on the box or a variable's contents — nothing a peer
    is owed and something a third party on the timeline should not read. The class, and no more."""
    detail = stall_detail(OSError("[Errno 28] No space left on device: '/home/glm-5.2/.harness'"))
    assert detail == "last error: an internal fault (OSError); the details are in its log"


def test_a_long_vendor_error_is_capped_not_softened():
    detail = stall_detail(ProviderServerError("x" * 5_000, status_code=502))
    assert detail.startswith("last error: xxx")
    assert len(detail) <= len("last error: ") + STALL_DETAIL_CAP
