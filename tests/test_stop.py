"""A stopped wake is a stop: SIGTERM ends it as `stopped`, and no later wake resumes it (issue #798).

The router's `stop-wake` op (basecradle-router#386) SIGTERMs every process in a wake and SIGKILLs
what is left about two seconds later. Before this issue the harness trapped nothing, so a stop looked
exactly like a crash, and the next organic wake did what the delivery guarantee tells it to do with a
crashed turn: it resumed it (`resuming … reason=died_mid_tool_chain`) and re-issued its creates. A
human stopped the wake, and the harness undid the stop — seen live on `@ping`, 2026-11-03.

These pin both halves and the line between them:

- **A stop is recorded as one.** The running turn's unanswered calls get the terminal `STOPPED`
  result — never `INTERRUPTED`, the sentinel the recovery re-issues creates from — and the turn ends
  on `STOPPED_NOTE`; every claim the wake holds is settled; `wake end` says `outcome=stopped
  signal=TERM`, which the NOC's `wake_failed` predicate (`outcome=error`) does not match.
- **The next wake neither resumes nor re-drives it**, and re-issues nothing — including when the
  stopping wake could not write the claim, because the turn's own note says what happened.
- **A crash is still a crash.** A wake killed with no trap (`SIGKILL`) leaves no note and no
  settled claim, and its turn is resumed exactly as before.
- **The process dies by the signal**, so `sudo` dies by -15 and the router's classification holds;
  a second SIGTERM (the router signals every process, and `sudo` relays) is ignored. These run a
  real process, because a signal's disposition is a property of one — and one of them stops a real
  SDK call blocked on its socket (issue #811).

In-process, a stop is raised the way the handler raises it: `WakeStopped` from wherever the main
thread happened to be — here, inside a tool, or inside the model call.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import httpx
import pytest
import respx

from basecradle_harness import Message, MessagesTool, Session, Tool, ToolCall
from basecradle_harness._session import INTERRUPTED, STOPPED, STOPPED_NOTE
from basecradle_harness._stop import WakeStopped
from basecradle_harness._wake import ClaimStore, SeenStore
from tests.conftest import plain
from tests.test_wake import (
    BC_URL,
    M0,
    REPLY,
    T0,
    TIMELINE_UUID,
    _posts,
    asset_page,
    build_wake,
    dashboard,
    event_page,
    message,
    page,
    serve_messages,
    task,
    task_page,
    timeline,
)

REPO = Path(__file__).resolve().parent.parent
BODY = "Can you look into the owl sightings for me?"


@pytest.fixture
def platform():
    """The respx-mocked platform — its own copy, as `test_resume` keeps one (see its note)."""
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


class _Script:
    """A brain that answers from a script; a `BaseException` in it is raised, not returned."""

    provider, model = "openai", "gpt-4o"

    def __init__(self, *replies) -> None:
        self.replies = list(replies)
        self.seen: list[list[Message]] = []

    def chat(self, messages, tools=None):
        self.seen.append(list(messages))
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply


class _Stopped(Tool):
    """A long tool the stop lands inside — what SIGTERM's handler raises, from where it was."""

    name = "survey"
    description = "Surveys the owls; takes a while."

    def __init__(self) -> None:
        self.runs = 0

    def run(self, **kwargs) -> str:
        self.runs += 1
        raise WakeStopped(signal.SIGTERM)


class _StoppedCreate(MessagesTool):
    """The `messages` tool, stopped while its create is in flight — the one unknowable outcome."""

    def run(self, **kwargs) -> str:
        if kwargs.get("action") == "create":
            raise WakeStopped(signal.SIGTERM)
        return super().run(**kwargs)


def _calls(*calls: tuple[str, str, dict]) -> Message:
    return Message.assistant(
        tool_calls=[ToolCall(id=i, name=name, arguments=args) for i, name, args in calls]
    )


def _transcript(home) -> list[dict]:
    path = home / "sessions" / f"timeline%3A{TIMELINE_UUID}.json"
    return json.loads(path.read_text())


def _lines(caplog) -> list[str]:
    return [plain(r.getMessage()) for r in caplog.records]


def _line(caplog, prefix: str) -> str:
    return next(m for m in _lines(caplog) if m.startswith(prefix))


def _stop_mid_tool(platform, home):
    """Wake one: the model posts, then calls a tool the stop lands inside."""
    serve_messages(platform, page(message(uuid=M0, body=BODY)))
    brain = _Script(
        _calls(
            ("c1", "messages", {"action": "create", "body": "On it."}),
            ("c2", "survey", {}),
        )
    )
    agent, _ = build_wake(home, brain, tools=[MessagesTool(), _Stopped()])
    with pytest.raises(WakeStopped):
        agent.wake()
    return brain


# --- the stopping wake records a stop -----------------------------------------------------


def test_a_wake_stopped_mid_tool_says_stopped_and_settles_what_it_held(platform, tmp_path, caplog):
    """The headline: `outcome=stopped signal=TERM`, the item settled, the turn ending on the note."""
    caplog.set_level("INFO", logger="basecradle_harness")
    _stop_mid_tool(platform, tmp_path)

    end = _line(caplog, "wake end")
    assert "outcome=stopped" in end and "signal=TERM" in end
    # The NOC's `wake_failed` reads `wake end .*outcome=error`; a stop is not a failure.
    assert re.search(r"wake end .*outcome=error", end) is None
    stopped = _line(caplog, "wake stopped")
    assert f"item={M0}" in stopped and "kind=messages" in stopped
    assert "reason=signal" in stopped and "signal=TERM" in stopped

    claim = ClaimStore(tmp_path).read(TIMELINE_UUID, M0, kind="messages")
    assert claim is not None and claim.phase == "abandoned"
    assert claim.reason == "stopped: the wake received SIGTERM"

    turns = _transcript(tmp_path)
    results = {t["tool_call_id"]: t["content"] for t in turns if t["role"] == "tool"}
    assert results["c2"] == STOPPED  # the call the stop landed in: terminal, never INTERRUPTED
    assert results["c1"] != STOPPED  # the post that finished before it keeps its real result
    assert turns[-1] == {"role": "system", "content": STOPPED_NOTE}


def test_the_next_wake_does_not_resume_a_stopped_turn(platform, tmp_path, caplog):
    """The bug, closed: no `resuming`, no model call, no second post, no re-run of the tool."""
    _stop_mid_tool(platform, tmp_path)
    assert _posts(platform) == ["On it."]
    caplog.clear()
    caplog.set_level("INFO", logger="basecradle_harness")

    serve_messages(platform, page(message(uuid=M0, body=BODY)))
    survey = _Stopped()
    second, brain = build_wake(tmp_path, _Script(), tools=[MessagesTool(), survey])
    assert second.wake() == []

    assert brain.seen == []  # nothing to finish, so the model is never called
    assert survey.runs == 0
    assert _posts(platform) == ["On it."]
    assert not any(
        line.startswith(("resuming", "re-driving", "re-issued")) for line in _lines(caplog)
    )
    assert "outcome=ok" in _line(caplog, "wake end")


def test_a_create_the_stop_interrupted_is_not_re_issued(platform, tmp_path, caplog):
    """A stop inside the create itself: its outcome is unknown, and the next wake leaves it so.

    Before #798 the next wake re-issued it under the dead wake's key — the platform's dedupe made
    that safe against a double post, but it was still the harness finishing work a human stopped.
    """
    serve_messages(platform, page(message(uuid=M0, body=BODY)))
    brain = _Script(_calls(("c1", "messages", {"action": "create", "body": "On it."})))
    first, _ = build_wake(tmp_path, brain, tools=[_StoppedCreate()])
    with pytest.raises(WakeStopped):
        first.wake()
    results = [t["content"] for t in _transcript(tmp_path) if t["role"] == "tool"]
    assert results == [STOPPED]

    caplog.set_level("INFO", logger="basecradle_harness")
    serve_messages(platform, page(message(uuid=M0, body=BODY)))
    second, brain = build_wake(tmp_path, _Script(), tools=[MessagesTool()])
    second.wake()

    assert _posts(platform) == []
    assert brain.seen == []
    assert not any(line.startswith("re-issued") for line in _lines(caplog))


def test_a_wake_stopped_inside_the_model_call_is_not_re_driven(platform, tmp_path, caplog):
    """A stop before any tool ran is still a stop: the item is settled, never answered later."""
    serve_messages(platform, page(message(uuid=M0, body=BODY)))
    first, _ = build_wake(tmp_path, _Script(WakeStopped(signal.SIGTERM)))
    with pytest.raises(WakeStopped):
        first.wake()
    assert _transcript(tmp_path)[-1] == {"role": "system", "content": STOPPED_NOTE}

    caplog.set_level("INFO", logger="basecradle_harness")
    serve_messages(platform, page(message(uuid=M0, body=BODY)))
    second, brain = build_wake(tmp_path, _Script())
    second.wake()

    assert brain.seen == []
    assert _posts(platform) == []
    assert not any(line.startswith("re-driving") for line in _lines(caplog))


def test_a_stopped_task_is_recorded_handled_and_never_re_run(platform, tmp_path, caplog):
    """A task has no cursor — its queue is the platform's `activated` list — so settling its claim
    as stopped must also get it into the seen-set, or it would be re-offered on every wake.
    """
    platform.get("/tasks").mock(
        return_value=httpx.Response(
            200, json=task_page(task(uuid=T0, instructions="Survey the owls, then report."))
        )
    )
    platform.get("/messages").mock(return_value=httpx.Response(200, json=page()))
    brain = _Script(_calls(("c1", "survey", {})))
    first, _ = build_wake(tmp_path, brain, tools=[MessagesTool(), _Stopped()])
    with pytest.raises(WakeStopped):
        first.wake()
    assert ClaimStore(tmp_path).read(TIMELINE_UUID, T0, kind="tasks").phase == "abandoned"

    caplog.set_level("INFO", logger="basecradle_harness")
    survey = _Stopped()
    second, brain = build_wake(tmp_path, _Script(), tools=[MessagesTool(), survey])
    second.wake()

    assert brain.seen == [] and survey.runs == 0
    assert T0 in SeenStore(tmp_path).all(TIMELINE_UUID, kind="tasks")
    assert not any(line.startswith(("resuming", "re-driving")) for line in _lines(caplog))


def test_a_stopped_turn_whose_claim_could_not_be_settled_is_read_off_its_note(
    platform, tmp_path, caplog, monkeypatch
):
    """The fallback: the claim write failed, so the next wake reads the turn's own stop note.

    The claim and the transcript are independent writes inside the grace window, and either can
    fail. The note is the transcript's record that the turn was stopped, so the claim left in
    flight is settled as stopped by the next wake rather than resumed as a crash.
    """

    def refused(self, *args, **kwargs):
        raise OSError(28, "No space left on device")

    with monkeypatch.context() as patch:
        patch.setattr(ClaimStore, "stop", refused)
        _stop_mid_tool(platform, tmp_path)
    claim = ClaimStore(tmp_path).read(TIMELINE_UUID, M0, kind="messages")
    assert claim is not None and claim.phase == "in-flight"  # the stop could not settle it

    caplog.set_level("INFO", logger="basecradle_harness")
    serve_messages(platform, page(message(uuid=M0, body=BODY)))
    second, brain = build_wake(tmp_path, _Script(), tools=[MessagesTool(), _Stopped()])
    second.wake()

    assert brain.seen == []
    assert _posts(platform) == ["On it."]
    stopped = _line(caplog, "wake stopped")
    assert f"item={M0}" in stopped and "reason=stop_note" in stopped
    assert not any(line.startswith("resuming") for line in _lines(caplog))
    claim = ClaimStore(tmp_path).read(TIMELINE_UUID, M0, kind="messages")
    assert claim is not None and claim.phase == "abandoned"


def test_a_stop_that_lands_outside_the_engine_still_leaves_the_note(
    platform, tmp_path, caplog, monkeypatch
):
    """The user turn is on disk and the model not yet called — `_drive`'s own handler never ran.

    The wake writes the note itself (`_note_stop`), so even with the claim write refused the next
    wake settles the item as stopped rather than re-driving it as `died_before_model`.
    """

    def stopped(*args, **kwargs):
        raise WakeStopped(signal.SIGTERM)

    def refused(self, *args, **kwargs):
        raise OSError(28, "No space left on device")

    serve_messages(platform, page(message(uuid=M0, body=BODY)))
    with monkeypatch.context() as patch:
        patch.setattr("basecradle_harness._session.log_context_attribution", stopped)
        patch.setattr(ClaimStore, "stop", refused)
        first, _ = build_wake(tmp_path, _Script())
        with pytest.raises(WakeStopped):
            first.wake()
    turns = _transcript(tmp_path)
    assert [t["role"] for t in turns if t["role"] != "system"] == ["user"]
    assert turns[-1] == {"role": "system", "content": STOPPED_NOTE}

    caplog.set_level("INFO", logger="basecradle_harness")
    serve_messages(platform, page(message(uuid=M0, body=BODY)))
    second, brain = build_wake(tmp_path, _Script())
    second.wake()

    assert brain.seen == [] and _posts(platform) == []
    assert "reason=stop_note" in _line(caplog, "wake stopped")
    assert not any(line.startswith("re-driving") for line in _lines(caplog))


def test_a_fault_settling_a_claim_never_replaces_the_stop(platform, tmp_path, monkeypatch):
    """Not only `OSError`: an undecodable claim raises `UnicodeDecodeError` from the read, and
    anything escaping `_stop_held` would replace the stop — a RED `wake failed`, exit 1, not -15.
    """

    def undecodable(self, *args, **kwargs):
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

    with monkeypatch.context() as patch:
        patch.setattr(ClaimStore, "stop", undecodable)
        _stop_mid_tool(platform, tmp_path)  # raises WakeStopped, not UnicodeDecodeError


def test_a_stop_settles_the_turn_an_earlier_pass_left_in_flight(platform, tmp_path):
    """A repass (#596) takes a fresh claims identity; what an earlier pass holds is still the wake's.

    The turn pass 1 left cut off is in flight under pass 1's identity, and a stop that lands in
    pass 2 before the recovery has taken it over must settle it too — or the next wake resumes it.
    """
    agent, _ = build_wake(tmp_path, _Script())
    ClaimStore(tmp_path, wake="pass-1").claim(TIMELINE_UUID, M0, kind="messages")
    agent.claims.wake = "pass-2"
    agent._wake_ids = {"pass-1", "pass-2"}

    agent._stop_held(WakeStopped(signal.SIGTERM))

    assert ClaimStore(tmp_path).read(TIMELINE_UUID, M0, kind="messages").phase == "abandoned"


# --- a crash is still a crash ---------------------------------------------------------------


def test_a_kill_with_no_trap_still_resumes_as_before(platform, tmp_path, caplog, monkeypatch):
    """`SIGKILL` runs no handler: no note, no settled claim, and the next wake resumes the turn.

    Simulated as `test_resume` does: the closing save in `Session`'s `finally` never happens (a
    `finally` does not run on a signal), and the tool dies with something that is not a stop — so
    none of #798's bookkeeping runs, exactly as on the box.
    """

    class _Killed(BaseException):
        """The process ending under the tool: no `except Exception` catches it, as none would."""

    class _Crashes(Tool):
        name = "survey"
        description = "Surveys the owls; takes a while."

        def run(self, **kwargs) -> str:
            raise _Killed("SIGKILL: the box went down mid-tool")

    serve_messages(platform, page(message(uuid=M0, body=BODY)))
    brain = _Script(_calls(("c1", "survey", {})))
    with monkeypatch.context() as patch:
        patch.setattr(Session, "_persist", lambda self, *, masking: None)
        first, _ = build_wake(tmp_path, brain, tools=[MessagesTool(), _Crashes()])
        with pytest.raises(_Killed):
            first.wake()
    assert not any(t.get("content") == STOPPED_NOTE for t in _transcript(tmp_path))

    caplog.set_level("INFO", logger="basecradle_harness")
    serve_messages(platform, page(message(uuid=M0, body=BODY)))
    finisher = _Script(Message.assistant(content="Picked it back up."))
    second, _ = build_wake(tmp_path, finisher, tools=[MessagesTool(), _Crashes()])
    second.wake()

    assert "reason=died_mid_tool_chain" in _line(caplog, "resuming")
    healed = next(m for m in finisher.seen[0] if m.role == "tool" and m.tool_call_id == "c1")
    assert healed.content == INTERRUPTED


# --- the claim a stop settles is exactly this wake's ----------------------------------------


def test_a_stop_settles_only_claims_this_wake_holds(tmp_path):
    """Its own claims — by any identity it used, or behind a take-over it won — and nothing else."""
    store = ClaimStore(tmp_path, wake="this-wake")
    store.claim(TIMELINE_UUID, "ours", kind="messages")
    rival = ClaimStore(tmp_path, wake="a-rival-wake")
    rival.claim(TIMELINE_UUID, "theirs", kind="messages")
    earlier = ClaimStore(tmp_path, wake="this-wake-pass-1")
    earlier.claim(TIMELINE_UUID, "an-earlier-pass", kind="messages")
    store.claim(TIMELINE_UUID, "settled", kind="messages")
    store.commit(TIMELINE_UUID, "settled", kind="messages")

    owners = frozenset({"this-wake", "this-wake-pass-1"})
    stopped = {
        uuid
        for uuid in ("ours", "theirs", "an-earlier-pass", "settled", "never-claimed")
        if store.stop(TIMELINE_UUID, uuid, kind="messages", owners=owners, reason="stopped")
    }

    assert stopped == {"ours", "an-earlier-pass"}
    assert store.read(TIMELINE_UUID, "theirs", kind="messages").phase == "in-flight"
    assert store.read(TIMELINE_UUID, "settled", kind="messages").phase == "done"


def test_a_stop_follows_a_take_over_that_has_not_rewritten_the_claim_yet(tmp_path):
    """A take-over this wake won and was stopped inside still names the owner it took over from."""
    dead = ClaimStore(tmp_path, wake="a-wake-that-died")
    dead.claim(TIMELINE_UUID, M0, kind="messages")
    store = ClaimStore(tmp_path, wake="this-wake")
    path = store._path(TIMELINE_UUID, "messages", M0)
    token = store._token(path, M0, "a-wake-that-died")
    token.write_text(json.dumps({"phase": "in-flight", "pid": os.getpid(), "wake": "this-wake"}))

    assert store.stop(
        TIMELINE_UUID, M0, kind="messages", owners=frozenset({"this-wake"}), reason="stopped"
    )
    assert store.read(TIMELINE_UUID, M0, kind="messages").phase == "abandoned"


# --- the process dies by the signal -----------------------------------------------------------


def _child(script: str, home: Path) -> subprocess.Popen:
    env = {**os.environ, "NO_COLOR": "1", "PYTHONPATH": str(REPO)}
    return subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(script), str(home)],
        cwd=REPO,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _await_ready(child: subprocess.Popen) -> None:
    line = child.stdout.readline()
    assert line.strip() == "READY", child.communicate(timeout=30)


def test_the_entrypoint_dies_by_sigterm_and_ignores_a_second_one(tmp_path):
    """`die` re-raises the signal itself, so the exit is a death by -15, not a code that mimics it.

    Two SIGTERMs land while the handler's work is running — the router signals every process in the
    wake and `sudo` relays the one it receives — and the second must not raise into the first one's
    bookkeeping.
    """
    child = _child(
        """
        import signal, sys, time
        from basecradle_harness import _stop

        _stop.install()
        try:
            print("READY", flush=True)
            time.sleep(30)
        except _stop.WakeStopped as stop:
            print("STOPPING", flush=True)
            time.sleep(0.5)  # the relayed second SIGTERM lands here, and must be ignored
            print("BOOKKEEPING DONE", flush=True)
            _stop.die(stop)
        print("UNREACHABLE", flush=True)
        """,
        tmp_path,
    )
    _await_ready(child)
    child.send_signal(signal.SIGTERM)
    assert child.stdout.readline().strip() == "STOPPING"
    child.send_signal(signal.SIGTERM)
    out, err = child.communicate(timeout=30)

    assert child.returncode == -signal.SIGTERM, err
    assert out.strip() == "BOOKKEEPING DONE"


def test_a_real_wake_sigtermed_mid_tool_logs_stopped_and_dies_by_the_signal(tmp_path):
    """End to end, in a real process: `main()`, a real wake, a tool the SIGTERM lands in.

    The platform is respx-mocked inside the child and the brain is scripted — nothing leaves the
    box — but the signal, the handler, the unwind, the log line and the exit are all real.
    """
    child = _child(
        """
        import sys, time
        from pathlib import Path

        import httpx, respx

        from basecradle_harness import Message, MessagesTool, Tool, ToolCall
        from basecradle_harness import _wake
        from tests import test_wake as w

        home = Path(sys.argv[1])

        class Survey(Tool):
            name = "survey"
            description = "Surveys the owls; takes a while."

            def run(self, **kwargs):
                print("READY", flush=True)
                time.sleep(30)
                return "done"

        class Brain:
            provider, model = "openai", "gpt-4o"

            def chat(self, messages, tools=None):
                return Message.assistant(tool_calls=[ToolCall(id="c1", name="survey", arguments={})])

        with respx.mock(base_url=w.BC_URL, assert_all_called=False) as router:
            router.get("/users/dashboard").mock(return_value=httpx.Response(200, json=w.dashboard()))
            router.get(f"/timelines/{w.TIMELINE_UUID}").mock(
                return_value=httpx.Response(200, json=w.timeline())
            )
            for path, body in (
                ("/assets", w.asset_page()),
                ("/webhook_events", w.event_page()),
                ("/tasks", w.task_page()),
            ):
                router.get(path).mock(return_value=httpx.Response(200, json=body))
            w.serve_messages(router, w.page(w.message(uuid=w.M0, body="Look into the owls?")))
            agent, _ = w.build_wake(home, Brain(), tools=[MessagesTool(), Survey()])
            _wake.WakeAgent.from_env = classmethod(lambda cls, **kwargs: agent)
            sys.exit(_wake.main(["--timeline", w.TIMELINE_UUID]))
        """,
        tmp_path,
    )
    _await_ready(child)
    child.send_signal(signal.SIGTERM)
    _, err = child.communicate(timeout=60)

    assert child.returncode == -signal.SIGTERM, err
    end = next(line for line in err.splitlines() if "wake end" in line)
    assert "outcome=stopped" in end and "signal=TERM" in end
    assert any("wake stopped" in line and "reason=signal" in line for line in err.splitlines())
    claim = ClaimStore(tmp_path).read(TIMELINE_UUID, M0, kind="messages")
    assert claim is not None and claim.phase == "abandoned"


def test_a_real_wake_sigtermed_inside_the_model_call_logs_stopped_and_dies_by_the_signal(
    tmp_path,
):
    """End to end, in a real process, with the stop landing inside a **real** SDK call (issue #811).

    The brain is the shipped `OpenAIProvider` driving the real `openai` SDK over HTTPX2 to a
    loopback endpoint that accepts the request and never answers, so the main thread is blocked in
    a socket read when SIGTERM arrives: where `@jt`'s live stop landed, 0.8 s after `context
    attribution`. The handler must still reach the main thread there, unwind the SDK, and leave
    both lines and a settled claim before the process dies by -15.

    (What silenced `@jt` was not this: `runuser` caught a second TERM and SIGKILLed the harness
    about 3 ms after the first, before any handler could log — a launcher fault no in-process code
    can outrun, recorded on #811.)
    """
    child = _child(
        """
        import socket, sys, threading, time
        from pathlib import Path

        import httpx, respx

        from basecradle_harness import MessagesTool
        from basecradle_harness import _wake
        from basecradle_harness._openai import OpenAIProvider
        from tests import test_wake as w

        home = Path(sys.argv[1])
        endpoint = socket.socket()
        endpoint.bind(("127.0.0.1", 0))
        endpoint.listen()

        def accept_and_never_answer():
            connection, _ = endpoint.accept()
            connection.recv(65536)
            print("READY", flush=True)  # the request is on the wire; the SDK is reading
            time.sleep(60)

        threading.Thread(target=accept_and_never_answer, daemon=True).start()
        brain = OpenAIProvider(
            "gpt-4o",
            api_key="sk-proj-not-a-real-key-000000000000",
            base_url=f"http://127.0.0.1:{endpoint.getsockname()[1]}/v1",
        )
        with respx.mock(base_url=w.BC_URL, assert_all_called=False) as router:
            router.route(host="127.0.0.1").pass_through()
            router.get("/users/dashboard").mock(return_value=httpx.Response(200, json=w.dashboard()))
            router.get(f"/timelines/{w.TIMELINE_UUID}").mock(
                return_value=httpx.Response(200, json=w.timeline())
            )
            for path, body in (
                ("/assets", w.asset_page()),
                ("/webhook_events", w.event_page()),
                ("/tasks", w.task_page()),
            ):
                router.get(path).mock(return_value=httpx.Response(200, json=body))
            w.serve_messages(router, w.page(w.message(uuid=w.M0, body="Look into the owls?")))
            agent, _ = w.build_wake(home, brain, tools=[MessagesTool()])
            _wake.WakeAgent.from_env = classmethod(lambda cls, **kwargs: agent)
            sys.exit(_wake.main(["--timeline", w.TIMELINE_UUID]))
        """,
        tmp_path,
    )
    _await_ready(child)
    child.send_signal(signal.SIGTERM)
    _, err = child.communicate(timeout=60)

    assert child.returncode == -signal.SIGTERM, err
    lines = err.splitlines()
    assert any(line.startswith("INFO context attribution") for line in lines), err
    assert not any(line.startswith("INFO llm ") or " outcome=ok" in line for line in lines), err
    end = next(line for line in lines if "wake end" in line)
    assert "outcome=stopped" in end and "signal=TERM" in end
    assert any("wake stopped" in line and "reason=signal" in line for line in lines)
    claim = ClaimStore(tmp_path).read(TIMELINE_UUID, M0, kind="messages")
    assert claim is not None and claim.phase == "abandoned"
