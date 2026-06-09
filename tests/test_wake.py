"""The router-driven wake entrypoint, against a respx-mocked platform.

Wake mode is one-shot and per-event: a fresh process answers a timeline's unseen
messages and exits. These tests pin the two properties the poll loop never had to
guarantee — **idempotency across processes** (a persisted high-water mark; a
re-invocation replies to nothing) and **persistence of the conversation** (the
transcript survives across wakes) — plus the first-wake bootstrap rules and the
cost guarantee (no provider call when nothing is new).

Each "process" is a freshly constructed `WakeAgent` over the same `home`, so the
only thing that carries between them is what is written to disk. The cast is the
fixed fiction: Nova Digital (``nova``, AI) is the agent; John Doe (``john``).
"""

import json

import httpx
import pytest
import respx
from basecradle import BaseCradle

from basecradle_harness import Harness, MarkStore, Message, WakeAgent
from basecradle_harness._wake import main

BC_URL = "https://basecradle.com"
FAKE_TOKEN = "bc_uat_KqI8zFxkQ0OZ8vYwT7mWcVtR3nSdLpEa"

NOVA_UUID = "019e7750-66ee-79c8-ad8a-bbb6ea7c2bcc"  # the agent (me)
JOHN_UUID = "019e7750-66ee-7e50-9e54-3bf8c3d6a8f1"  # the human
TIMELINE_UUID = "019e7750-66ee-7f53-829f-13a8a710b6da"

# Well-formed UUIDv7 message ids, oldest → newest.
M0 = "019e7751-4a1b-7c2d-8e3f-1a2b3c4d5e6f"
M1 = "019e7752-5b2c-7d3e-9f40-2b3c4d5e6f70"
M2 = "019e7753-6c3d-7e4f-8051-3c4d5e6f7081"
M3 = "019e7754-7d4e-7e60-8172-4d5e6f708192"
REPLY = "019e7755-8e5f-7f70-9283-5e6f70819203"


# --- wire payload builders (mirrors test_basecradle_io) ----------------------


def message(*, uuid, body, mine=False):
    actor = (
        {"uuid": NOVA_UUID, "handle": "nova", "name": "Nova Digital", "kind": "ai"}
        if mine
        else {"uuid": JOHN_UUID, "handle": "john", "name": "John Doe", "kind": "human"}
    )
    return {
        "type": "message",
        "created_at": "2026-06-04T00:00:00.000Z",
        "user": actor,
        "timeline": {"uuid": TIMELINE_UUID},
        "content": {"uuid": uuid, "body": body},
    }


def page(*messages):
    return {"messages": list(messages), "next_cursor": None}


def dashboard():
    return {"identity": {"uuid": NOVA_UUID, "handle": "nova", "name": "Nova Digital", "kind": "ai"}}


def timeline():
    return {
        "timeline": {
            "uuid": TIMELINE_UUID,
            "name": "Incident response",
            "locked": False,
            "created_at": "2026-06-01T00:00:00.000Z",
            "updated_at": "2026-06-02T00:00:00.000Z",
            "owner": {"uuid": JOHN_UUID, "handle": "john", "name": "John Doe", "kind": "human"},
            "participants": [
                {"uuid": NOVA_UUID, "handle": "nova", "name": "Nova Digital", "kind": "ai"}
            ],
        },
        "items": [],
    }


class CountingProvider:
    """A canned brain that records every call, so we can assert when it is (not) used."""

    def __init__(self, text="Hello, John."):
        self.text = text
        self.prompts: list[str] = []
        self.last_messages: list = []

    def chat(self, messages, tools=None):
        self.last_messages = list(messages)
        self.prompts.append(messages[-1].content)
        return Message.assistant(content=self.text)


@pytest.fixture
def platform():
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
        yield router


def serve_messages(platform, *pages):
    """Drive the (newest-first) message list endpoint; one page per read."""
    platform.get("/messages").mock(side_effect=[httpx.Response(200, json=p) for p in pages])


def build_wake(home, provider=None, *, system_prompt=None, onboard=False, **kwargs):
    """A fresh WakeAgent over `home` — a stand-in for one router-spawned process."""
    provider = provider or CountingProvider()
    client = BaseCradle(token=FAKE_TOKEN)
    harness = Harness(provider, system_prompt=system_prompt, home=home)
    agent = WakeAgent(harness, timeline=TIMELINE_UUID, client=client, onboard=onboard, **kwargs)
    return agent, provider


# --- the core contract: reply once, then be idempotent -----------------------


def test_single_new_message_gets_one_reply(platform, tmp_path):
    """First wake on a timeline with one message replies to it exactly once."""
    serve_messages(platform, page(message(uuid=M0, body="What's the status?")))
    agent, provider = build_wake(tmp_path)

    posted = agent.wake()

    assert len(posted) == 1
    assert provider.prompts == ["john: What's the status?"]
    sent = platform.post(f"/timelines/{TIMELINE_UUID}/messages").calls.last.request
    assert json.loads(sent.content) == {"message": {"body": "Hello, John."}}


def test_reinvocation_replies_to_nothing(platform, tmp_path):
    """A second process (same home) sees the persisted mark and answers nothing."""
    # First wake: one message, one reply. Mark is persisted under `home`.
    serve_messages(platform, page(message(uuid=M0, body="hi")))
    first, first_provider = build_wake(tmp_path)
    assert len(first.wake()) == 1

    # Second wake: a brand-new process, same timeline state — nothing newer than M0.
    serve_messages(platform, page(message(uuid=M0, body="hi")))
    second, second_provider = build_wake(tmp_path)

    assert second.wake() == []
    assert second_provider.prompts == []  # the model was never consulted
    assert not platform.post(f"/timelines/{TIMELINE_UUID}/messages").calls[1:]  # no 2nd post


def test_no_new_messages_makes_no_provider_call(platform, tmp_path):
    """With a mark already at the newest message, a wake is free: no model call."""
    MarkStore(tmp_path).set(TIMELINE_UUID, M0)  # pretend a prior wake already saw M0
    serve_messages(platform, page(message(uuid=M0, body="old")))
    agent, provider = build_wake(tmp_path)

    assert agent.wake() == []
    assert provider.prompts == []


def test_new_message_after_a_mark_is_answered(platform, tmp_path):
    """Steady state: a message newer than the persisted mark gets a reply."""
    MarkStore(tmp_path).set(TIMELINE_UUID, M0)
    serve_messages(
        platform, page(message(uuid=M1, body="new question"), message(uuid=M0, body="old"))
    )
    agent, provider = build_wake(tmp_path)

    posted = agent.wake()

    assert len(posted) == 1
    assert provider.prompts == ["john: new question"]
    assert MarkStore(tmp_path).get(TIMELINE_UUID) == M1  # mark advanced


def test_multiple_unseen_messages_answered_oldest_first(platform, tmp_path):
    MarkStore(tmp_path).set(TIMELINE_UUID, M0)
    serve_messages(
        platform,
        page(
            message(uuid=M2, body="second"),
            message(uuid=M1, body="first"),
            message(uuid=M0, body="old"),
        ),
    )
    agent, provider = build_wake(tmp_path)

    posted = agent.wake()

    assert len(posted) == 2
    assert provider.prompts == ["john: first", "john: second"]  # chronological
    assert MarkStore(tmp_path).get(TIMELINE_UUID) == M2


# --- idempotency across a crash / retry mid-batch ----------------------------


def test_mark_advances_after_each_reply(platform, tmp_path):
    """The mark is persisted per message, so a retry resumes without duplicating.

    The provider raises on the *second* message, mimicking a process that dies
    mid-batch. The mark must already reflect the first message, so the retry only
    has the second left to answer.
    """
    MarkStore(tmp_path).set(TIMELINE_UUID, M0)
    serve_messages(
        platform,
        page(
            message(uuid=M2, body="second"),
            message(uuid=M1, body="first"),
            message(uuid=M0, body="old"),
        ),
    )

    class DiesOnSecond:
        def __init__(self):
            self.calls = 0

        def chat(self, messages, tools=None):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("boom mid-batch")
            return Message.assistant(content="ok")

    agent, _ = build_wake(tmp_path, DiesOnSecond())
    with pytest.raises(RuntimeError):
        agent.wake()

    # The first message was answered and its mark persisted before the crash.
    assert MarkStore(tmp_path).get(TIMELINE_UUID) == M1


# --- the conversation persists across wakes ----------------------------------


def test_transcript_persists_across_wakes(platform, tmp_path):
    """A second wake reloads the first's transcript instead of starting blank."""
    serve_messages(platform, page(message(uuid=M0, body="remember Ruby")))
    first, _ = build_wake(tmp_path)
    first.wake()

    serve_messages(
        platform,
        page(message(uuid=M1, body="what did I say?"), message(uuid=M0, body="remember Ruby")),
    )
    second, provider = build_wake(tmp_path)
    second.wake()

    # The model saw the earlier exchange (loaded from disk) in front of the new turn.
    roles_and_text = [(m.role, m.content) for m in provider.last_messages]
    assert ("user", "john: remember Ruby") in roles_and_text
    assert ("assistant", "Hello, John.") in roles_and_text
    assert roles_and_text[-1] == ("user", "john: what did I say?")


# --- first-wake bootstrap rules ----------------------------------------------


def test_bootstrap_replies_to_everything_since_our_last_post(platform, tmp_path):
    """Cutover from poll mode: reply to all messages after our own latest message."""
    serve_messages(
        platform,
        page(
            message(uuid=M3, body="and another"),
            message(uuid=M2, body="a follow-up"),
            message(uuid=M1, body="my last reply", mine=True),  # our high-water footprint
            message(uuid=M0, body="ancient history"),
        ),
    )
    agent, provider = build_wake(tmp_path)

    posted = agent.wake()

    assert len(posted) == 2
    assert provider.prompts == ["john: a follow-up", "john: and another"]
    # M0 and our own M1 are context, not replied to; the mark is the true newest.
    assert MarkStore(tmp_path).get(TIMELINE_UUID) == M3


def test_bootstrap_with_trigger_replies_from_the_trigger_forward(platform, tmp_path):
    """A named triggering message bounds the bootstrap precisely."""
    serve_messages(
        platform,
        page(
            message(uuid=M2, body="newest"),
            message(uuid=M1, body="the trigger"),
            message(uuid=M0, body="older context"),
        ),
    )
    agent, provider = build_wake(tmp_path)

    agent.wake(trigger=M1)

    assert provider.prompts == ["john: the trigger", "john: newest"]  # M1 forward
    assert MarkStore(tmp_path).get(TIMELINE_UUID) == M2


def test_bootstrap_fresh_join_replies_to_newest_only(platform, tmp_path):
    """Never having spoken here, reply to the message that woke us — not all history."""
    serve_messages(
        platform,
        page(
            message(uuid=M2, body="newest"),
            message(uuid=M1, body="middle"),
            message(uuid=M0, body="oldest"),
        ),
    )
    agent, provider = build_wake(tmp_path)

    posted = agent.wake()

    assert len(posted) == 1
    assert provider.prompts == ["john: newest"]  # only the newest, history is context
    assert MarkStore(tmp_path).get(TIMELINE_UUID) == M2


def test_bootstrap_seeds_history_as_context_before_replying(platform, tmp_path):
    """The backlog older than the reply set is in front of the model when it answers."""
    serve_messages(
        platform,
        page(message(uuid=M1, body="what did we decide?"), message(uuid=M0, body="we chose Ruby")),
    )
    agent, provider = build_wake(tmp_path)

    agent.wake()

    context = [(m.role, m.content) for m in provider.last_messages]
    assert context == [
        ("user", "john: we chose Ruby"),  # seeded backlog
        ("user", "john: what did we decide?"),  # the message being answered
    ]


def test_bootstrap_when_our_own_post_is_newest_replies_to_nothing(platform, tmp_path):
    """If the latest message is our own, there is nothing unseen — mark, don't reply."""
    serve_messages(platform, page(message(uuid=M0, body="my last word", mine=True)))
    agent, provider = build_wake(tmp_path)

    assert agent.wake() == []
    assert provider.prompts == []
    assert MarkStore(tmp_path).get(TIMELINE_UUID) == M0  # still marked, so we don't re-scan it


def test_bootstrap_empty_timeline_is_a_noop(platform, tmp_path):
    serve_messages(platform, page())
    agent, provider = build_wake(tmp_path)

    assert agent.wake() == []
    assert provider.prompts == []
    assert MarkStore(tmp_path).get(TIMELINE_UUID) is None  # nothing to mark


def test_does_not_reply_to_its_own_new_message(platform, tmp_path):
    """A new message authored by us advances the mark but draws no reply."""
    MarkStore(tmp_path).set(TIMELINE_UUID, M0)
    serve_messages(
        platform,
        page(message(uuid=M1, body="my own post", mine=True), message(uuid=M0, body="old")),
    )
    agent, provider = build_wake(tmp_path)

    assert agent.wake() == []
    assert provider.prompts == []
    assert MarkStore(tmp_path).get(TIMELINE_UUID) == M1


# --- the session key + shared memory -----------------------------------------


def test_wake_uses_a_timeline_scoped_session(platform, tmp_path):
    """Each wake runs the `timeline:<uuid>` session — one identity, channel-keyed."""
    serve_messages(platform, page(message(uuid=M0, body="hi")))
    agent, _ = build_wake(tmp_path)
    agent.wake()

    assert agent.source == f"timeline:{TIMELINE_UUID}"
    assert f"timeline:{TIMELINE_UUID}" in agent.harness.sessions


# --- construction guards -----------------------------------------------------


def test_wake_requires_a_home_for_persistence(platform):
    """Without a home (or explicit marks) the mark cannot persist — fail fast."""
    harness = Harness(CountingProvider())  # no home
    with pytest.raises(ValueError, match="HARNESS_HOME"):
        WakeAgent(harness, timeline=TIMELINE_UUID, client=BaseCradle(token=FAKE_TOKEN))


def test_negative_context_messages_is_rejected(platform, tmp_path):
    harness = Harness(CountingProvider(), home=tmp_path)
    with pytest.raises(ValueError):
        WakeAgent(
            harness,
            timeline=TIMELINE_UUID,
            client=BaseCradle(token=FAKE_TOKEN),
            context_messages=-1,
        )


# --- the `basecradle-harness-wake` CLI ---------------------------------------


@pytest.fixture
def wake_env(monkeypatch, tmp_path):
    """The full environment the router would source before invoking the CLI."""
    monkeypatch.setenv("BASECRADLE_TOKEN", FAKE_TOKEN)
    monkeypatch.setenv("AI_PROVIDER_MODEL", "gpt-4o")
    monkeypatch.setenv("AI_PROVIDER_API_KEY", "sk-test-key")
    monkeypatch.setenv("HARNESS_HOME", str(tmp_path))
    monkeypatch.setenv("HARNESS_ONBOARD", "0")
    monkeypatch.delenv("BASECRADLE_TIMELINE", raising=False)
    monkeypatch.delenv("BASECRADLE_MESSAGE", raising=False)
    return tmp_path


def _serve_openai_and_messages(platform, *pages):
    """A plain-text model reply plus the message pages — enough for `main` to run live."""
    platform.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-wake",
                "object": "chat.completion",
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "On it."},
                        "finish_reason": "stop",
                    }
                ],
            },
        )
    )
    serve_messages(platform, *pages)


def test_main_processes_a_timeline_and_exits_zero(platform, wake_env):
    _serve_openai_and_messages(platform, page(message(uuid=M0, body="status?")))

    assert main(["--timeline", TIMELINE_UUID]) == 0
    assert platform.post(f"/timelines/{TIMELINE_UUID}/messages").called  # it replied


def test_main_is_idempotent_on_reinvocation(platform, wake_env):
    _serve_openai_and_messages(platform, page(message(uuid=M0, body="hi")))
    assert main(["--timeline", TIMELINE_UUID]) == 0

    # A second invocation (new "process", same HARNESS_HOME) replies to nothing.
    _serve_openai_and_messages(platform, page(message(uuid=M0, body="hi")))
    assert main(["--timeline", TIMELINE_UUID]) == 0
    assert platform.post(f"/timelines/{TIMELINE_UUID}/messages").call_count == 1


def test_main_reads_the_timeline_from_the_environment(platform, wake_env, monkeypatch):
    monkeypatch.setenv("BASECRADLE_TIMELINE", TIMELINE_UUID)
    _serve_openai_and_messages(platform, page(message(uuid=M0, body="hi")))

    assert main([]) == 0  # no --timeline flag; BASECRADLE_TIMELINE supplies it


def test_main_without_a_timeline_errors(wake_env):
    """No --timeline and no BASECRADLE_TIMELINE → argparse exits non-zero."""
    with pytest.raises(SystemExit) as exit_info:
        main([])
    assert exit_info.value.code != 0


def test_main_returns_nonzero_when_home_is_missing(platform, wake_env, monkeypatch):
    """A hard config failure (no HARNESS_HOME) exits non-zero so the router reports it."""
    monkeypatch.delenv("HARNESS_HOME", raising=False)

    assert main(["--timeline", TIMELINE_UUID]) == 1


def test_main_returns_nonzero_on_missing_provider_config(platform, wake_env, monkeypatch):
    monkeypatch.delenv("AI_PROVIDER_MODEL", raising=False)

    assert main(["--timeline", TIMELINE_UUID]) == 1
