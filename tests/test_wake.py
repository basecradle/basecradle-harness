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

import base64
import hmac
import json
import re
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from types import SimpleNamespace

import httpx
import pytest
import respx
from basecradle import BaseCradle

from basecradle_harness import (
    BreakerDecision,
    ClaimStore,
    Harness,
    MarkStore,
    Message,
    ReadPacer,
    SeenStore,
    WakeAgent,
    WakeBreaker,
    install,
)
from basecradle_harness._basecradle import _incoming_text, _parse_created_at
from basecradle_harness._messages import ToolCall
from basecradle_harness._wake import (
    _BREAKER_RESET_ALERT,
    _BREAKER_TRIP_ALERT,
    _activated_task_text,
    _incoming_asset_text,
    _incoming_event_text,
    _now_line,
    _pace_chars_per_sec_from_env,
    _pace_enabled_from_env,
    _pace_floor_seconds_from_env,
    main,
    resolved_config,
)

BC_URL = "https://basecradle.com"
FAKE_TOKEN = "bc_uat_KqI8zFxkQ0OZ8vYwT7mWcVtR3nSdLpEa"

# A tiny stand-in for a real image blob: the asset builder points every asset's blob
# url at `{BC_URL}/blobs/<uuid>`, and the `platform` fixture serves these bytes there,
# so an asset wake's eager image fetch (perception) succeeds without a per-test mock.
PNG_BYTES = b"\x89PNG\r\n\x1a\n fake pixels"

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


# Well-formed UUIDv7 webhook-event and endpoint ids, oldest → newest.
EP = "019e7760-1111-7aaa-8bbb-1c2d3e4f5061"
E0 = "019e7761-2222-7bbb-8ccc-2d3e4f506172"
E1 = "019e7762-3333-7ccc-8ddd-3e4f50617283"


def event(*, uuid, payload, content_type="application/json"):
    return {
        "type": "webhook_event",
        "created_at": "2026-06-04T00:00:00.000Z",
        "timeline": {"uuid": TIMELINE_UUID},
        "webhook_endpoint": {"uuid": EP},
        "content": {
            "uuid": uuid,
            "content_type": content_type,
            "headers": {"x-source": "github"},
            "payload": payload,
            "ingest_token_at_receipt": "tok_abc",
        },
    }


def event_page(*events):
    return {"webhook_events": list(events), "next_cursor": None}


# Well-formed UUIDv7 task ids.
T0 = "019e7770-5555-7eee-8fff-506172839405"
T1 = "019e7771-6666-7fff-8aaa-617283940516"


def task(*, uuid, instructions, status="activated", activate_at="2026-06-11T06:00:00+00:00"):
    return {
        "type": "task",
        "created_at": "2026-06-10T00:00:00.000Z",
        "user": {"uuid": NOVA_UUID, "handle": "nova", "name": "Nova Digital", "kind": "ai"},
        "timeline": {"uuid": TIMELINE_UUID},
        "content": {
            "uuid": uuid,
            "instructions": instructions,
            "activate_at": activate_at,
            "status": status,
        },
    }


def task_page(*tasks):
    return {"tasks": list(tasks), "next_cursor": None}


# Well-formed UUIDv7 asset ids.
A0 = "019e7780-7777-7aaa-8bbb-728394051627"
A1 = "019e7781-8888-7bbb-8ccc-839405162738"


def asset(*, uuid, filename="photo.png", content_type="image/png", description="", mine=False):
    actor = (
        {"uuid": NOVA_UUID, "handle": "nova", "name": "Nova Digital", "kind": "ai"}
        if mine
        else {"uuid": JOHN_UUID, "handle": "john", "name": "John Doe", "kind": "human"}
    )
    return {
        "type": "asset",
        "created_at": "2026-06-04T00:00:00.000Z",
        "user": actor,
        "timeline": {"uuid": TIMELINE_UUID},
        "content": {
            "uuid": uuid,
            "description": description,
            "file": {
                "filename": filename,
                "byte_size": 2048,
                "content_type": content_type,
                "checksum": "Yp9p9C8m6Xv2qS1nKQ0r3w==",
                "url": f"{BC_URL}/blobs/{uuid}",
            },
        },
    }


def asset_page(*assets):
    return {"assets": list(assets), "next_cursor": None}


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
        # A snapshot of the images on the last turn at *chat time* — captured before the
        # session evicts the pixels, so a test can assert an asset image was actually
        # presented to the model (the live object's `.images` is emptied after the turn).
        self.last_images: list = []

    def chat(self, messages, tools=None):
        self.last_messages = list(messages)
        self.last_images = list(messages[-1].images)
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
        # By default a timeline has no posted assets, no inbound webhook deliveries,
        # and no activated tasks, so a wake's reconciliation of them is a clean no-op.
        # Tests that exercise those paths override these with the matching `serve_*`.
        router.get("/assets").mock(return_value=httpx.Response(200, json=asset_page()))
        router.get("/webhook_events").mock(return_value=httpx.Response(200, json=event_page()))
        router.get("/tasks").mock(return_value=httpx.Response(200, json=task_page()))
        # A peer's posted image is fetched and shown to the model on wake (perception),
        # so every asset's blob url resolves to image bytes by default. A test exercising
        # a download failure overrides this with a 5xx/connection error.
        router.get(path__regex=r"^/blobs/").mock(
            return_value=httpx.Response(200, content=PNG_BYTES)
        )
        yield router


def serve_messages(platform, *pages):
    """Drive the (newest-first) message list endpoint; one page per read."""
    platform.get("/messages").mock(side_effect=[httpx.Response(200, json=p) for p in pages])


def serve_events(platform, *pages):
    """Drive the (newest-first) webhook-event list endpoint; one page per read."""
    platform.get("/webhook_events").mock(side_effect=[httpx.Response(200, json=p) for p in pages])


def serve_tasks(platform, *pages):
    """Drive the (newest-first) task list endpoint; one page per read."""
    platform.get("/tasks").mock(side_effect=[httpx.Response(200, json=p) for p in pages])


def serve_assets(platform, *pages):
    """Drive the (newest-first) asset list endpoint; one page per read."""
    platform.get("/assets").mock(side_effect=[httpx.Response(200, json=p) for p in pages])


def serve_dashboard_md(platform, text="# Dashboard\n\nTrust is mutual at the gate.\n"):
    """Drive the live `dashboard.md` primer the persistent brief fetches each wake."""
    return platform.get("/users/dashboard.md").mock(return_value=httpx.Response(200, text=text))


def _brief_turns(agent):
    """The persistent-brief system turns in the agent's live transcript (by their header)."""
    history = agent.harness.session(agent.source).history
    return [m for m in history if m.role == "system" and "How to operate here" in (m.content or "")]


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
    assert provider.prompts == ["[2026-06-04T00:00:00.000Z] john: What's the status?"]
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
    assert provider.prompts == ["[2026-06-04T00:00:00.000Z] john: new question"]
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
    assert provider.prompts == [
        "[2026-06-04T00:00:00.000Z] john: first",
        "[2026-06-04T00:00:00.000Z] john: second",
    ]  # chronological
    assert MarkStore(tmp_path).get(TIMELINE_UUID) == M2


# --- idempotency across a crash / retry mid-batch ----------------------------


def test_a_mid_batch_crash_does_not_reprocess_the_crashed_message(platform, tmp_path):
    """B3: claim-first — a forced mid-wake failure advances the mark over the crashed
    message, so it is never reprocessed (no model turn re-burned, no tool action re-fired).

    The provider raises on the *second* message, mimicking a process that dies mid-batch.
    Because each message is claimed and marked seen *before* it is acted on, both the
    answered first message and the crashed second leave the mark at the true newest — a
    retry replies to nothing rather than re-answering M2 on every later wake (the live
    reprocess loop). At-most-once: a one-time dropped reply beats a backlog re-answered.
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

    # Both messages were claimed and marked seen before acting, so the mark is the newest —
    # the crashed M2 will not be re-answered on the next wake.
    assert MarkStore(tmp_path).get(TIMELINE_UUID) == M2
    assert ClaimStore(tmp_path).claim(TIMELINE_UUID, M2, kind="messages") is False  # M2 was claimed

    # A retry (fresh process, same home) sees the advanced mark and replies to nothing.
    serve_messages(platform, page(message(uuid=M2, body="second"), message(uuid=M1, body="first")))
    retry, retry_provider = build_wake(tmp_path)
    assert retry.wake() == []
    assert retry_provider.prompts == []


# --- B2: a wake never crashes on an SDK / engine error -----------------------


def _locked_problem():
    """An RFC 9457 problem doc for a locked timeline — what the post is refused with."""
    return {
        "status": 403,
        "code": "timeline_locked",
        "title": "Timeline Locked",
        "detail": "This timeline is locked and is not accepting new content.",
    }


def test_a_locked_timeline_post_degrades_instead_of_crashing(platform, tmp_path):
    """B2: a reply refused by a locked timeline degrades to an in-conversation note, never
    a crash. The model still ran, the message is still marked seen (no reprocess loop), and
    the wake completes cleanly (exit 0)."""
    MarkStore(tmp_path).set(TIMELINE_UUID, M0)
    serve_messages(platform, page(message(uuid=M1, body="hi"), message(uuid=M0, body="old")))
    platform.post(f"/timelines/{TIMELINE_UUID}/messages").mock(
        return_value=httpx.Response(403, json=_locked_problem())
    )
    agent, provider = build_wake(tmp_path)

    posted = agent.wake()  # must not raise

    assert posted == []  # the reply could not be delivered
    assert provider.prompts == ["[2026-06-04T00:00:00.000Z] john: hi"]  # the model still ran
    assert MarkStore(tmp_path).get(TIMELINE_UUID) == M1  # marked seen despite the failed post
    # The failed delivery is recorded as a note in the transcript, so the record stays honest.
    notes = [t.content for t in agent.harness.session(agent.source).history if t.role == "system"]
    assert any("Couldn't post that reply" in n for n in notes)


def test_main_returns_zero_on_a_locked_timeline(platform, wake_env):
    """B2 end to end: the entrypoint exits 0 on a locked timeline — no traceback escapes.

    The model replies, but the timeline refuses the post (`timeline_locked`); the degrade
    keeps the wake clean and the exit code 0, so the router sees success, not a crash."""
    _serve_openai_and_messages(platform, page(message(uuid=M0, body="status?")))
    platform.post(f"/timelines/{TIMELINE_UUID}/messages").mock(
        return_value=httpx.Response(403, json=_locked_problem())
    )

    assert main(["--timeline", TIMELINE_UUID]) == 0


def test_max_steps_degrades_to_a_graceful_reply(platform, tmp_path):
    """B2: the engine's step cap (EngineError) degrades to a short "I got stuck" reply that
    is posted, rather than crashing the wake. The message is still marked seen."""
    serve_messages(platform, page(message(uuid=M0, body="do something complicated")))

    class LoopingProvider:
        """A brain that always asks for a tool, so the loop never settles → max_steps."""

        def __init__(self):
            self.prompts = []

        def chat(self, messages, tools=None):
            self.prompts.append(messages[-1].content)
            return Message.assistant(
                content=None,
                tool_calls=[ToolCall(id="call_1", name="memory", arguments={"action": "list"})],
            )

    client = BaseCradle(token=FAKE_TOKEN)
    harness = Harness(LoopingProvider(), home=tmp_path, max_steps=2)
    agent = WakeAgent(harness, timeline=TIMELINE_UUID, client=client, onboard=False)

    posted = agent.wake()  # must not raise

    assert len(posted) == 1
    sent = platform.post(f"/timelines/{TIMELINE_UUID}/messages").calls.last.request
    assert "got stuck" in json.loads(sent.content)["message"]["body"]
    assert agent.marks.get(TIMELINE_UUID) == M0  # still marked seen — no reprocess


# --- B8: concurrent wakes handle a message exactly once -----------------------


def test_two_concurrent_wakes_reply_to_a_message_exactly_once(platform, tmp_path):
    """B8: an upload firing asset.created + message.created spawns two wakes that both see
    the same unseen message. The atomic claim makes exactly one of them act — the second,
    losing the claim, replies to nothing. (The two wakes are simulated by claiming M1 with
    the first agent's ClaimStore before the second runs — the post-claim race resolved.)"""
    MarkStore(tmp_path).set(TIMELINE_UUID, M0)
    serve_messages(platform, page(message(uuid=M1, body="new"), message(uuid=M0, body="old")))

    # The first wake wins the claim and replies.
    first, first_provider = build_wake(tmp_path)
    assert len(first.wake()) == 1
    assert first_provider.prompts == ["[2026-06-04T00:00:00.000Z] john: new"]

    # A second, concurrent wake (fresh process) had already read mark=M0 and computed M1 as
    # unseen — but the mark has since advanced AND M1 is claimed, so it acts on nothing.
    MarkStore(tmp_path).set(TIMELINE_UUID, M0)  # simulate it still holding the stale mark
    serve_messages(platform, page(message(uuid=M1, body="new"), message(uuid=M0, body="old")))
    second, second_provider = build_wake(tmp_path)

    assert second.wake() == []  # the claim blocked the duplicate
    assert second_provider.prompts == []  # the model was never consulted a second time


def test_claim_store_is_atomic_exactly_once(tmp_path):
    """The claim primitive: the first claim of (kind, timeline, uuid) wins, any later one
    loses — the exclusive-create that makes concurrent wakes mutually exclusive."""
    claims = ClaimStore(tmp_path)
    assert claims.claim(TIMELINE_UUID, M1, kind="messages") is True
    assert claims.claim(TIMELINE_UUID, M1, kind="messages") is False  # already owned
    # A different uuid, or the same uuid under a different kind, is an independent claim.
    assert claims.claim(TIMELINE_UUID, M2, kind="messages") is True
    assert claims.claim(TIMELINE_UUID, M1, kind="assets") is True


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
    assert ("user", "[2026-06-04T00:00:00.000Z] john: remember Ruby") in roles_and_text
    assert ("assistant", "Hello, John.") in roles_and_text
    assert roles_and_text[-1] == ("user", "[2026-06-04T00:00:00.000Z] john: what did I say?")


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
    assert provider.prompts == [
        "[2026-06-04T00:00:00.000Z] john: a follow-up",
        "[2026-06-04T00:00:00.000Z] john: and another",
    ]
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

    assert provider.prompts == [
        "[2026-06-04T00:00:00.000Z] john: the trigger",
        "[2026-06-04T00:00:00.000Z] john: newest",
    ]  # M1 forward
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
    assert provider.prompts == [
        "[2026-06-04T00:00:00.000Z] john: newest"
    ]  # only the newest, history is context
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
        ("user", "[2026-06-04T00:00:00.000Z] john: we chose Ruby"),  # seeded backlog
        (
            "user",
            "[2026-06-04T00:00:00.000Z] john: what did we decide?",
        ),  # the message being answered
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


# --- inbound webhook deliveries (webhook_event.received) ---------------------


def test_first_wake_acts_on_the_triggering_event(platform, tmp_path):
    """The router wakes the agent on a delivery and names it; the agent acts on it."""
    serve_messages(platform, page())  # no messages on the timeline
    serve_events(platform, event_page(event(uuid=E0, payload='{"action":"opened"}')))
    agent, provider = build_wake(tmp_path)

    posted = agent.wake(event_trigger=E0)

    assert len(posted) == 1  # the agent perceived the delivery and replied
    assert len(provider.prompts) == 1
    assert "inbound webhook" in provider.prompts[0]
    assert '{"action":"opened"}' in provider.prompts[0]  # the payload reached the model
    assert MarkStore(tmp_path).get(TIMELINE_UUID, kind="webhook_events") == E0


def test_first_wake_without_a_trigger_acts_on_the_newest_delivery(platform, tmp_path):
    """THE BUG-2 REGRESSION: the router wakes a harness agent with the timeline uuid
    alone — it never passes `--event` — so a webhook wake arrives with no trigger. The
    first wake must still act on the delivery (the newest unseen one), not baseline it
    away. The old behavior dropped every first delivery; that is exactly what made
    `webhook_event.received` surface nothing live."""
    serve_messages(platform, page())
    serve_events(platform, event_page(event(uuid=E0, payload="PAPAYA-CLEAN-42")))
    agent, provider = build_wake(tmp_path)

    posted = agent.wake()  # no event_trigger — the real router contract

    assert len(posted) == 1  # the delivery was acted on autonomously
    assert "inbound webhook" in provider.prompts[0]
    assert "PAPAYA-CLEAN-42" in provider.prompts[0]  # the payload reached the model
    assert MarkStore(tmp_path).get(TIMELINE_UUID, kind="webhook_events") == E0  # and marked


def test_first_wake_without_a_trigger_acts_on_newest_event_only(platform, tmp_path):
    """A first event wake with no trigger acts on the newest unseen delivery and marks
    past the rest, so a fresh agent is bounded to one action, not a backlog replay."""
    serve_messages(platform, page())
    serve_events(
        platform,
        event_page(event(uuid=E1, payload="newest"), event(uuid=E0, payload="older")),
    )
    agent, provider = build_wake(tmp_path)

    posted = agent.wake()

    assert len(posted) == 1
    assert "newest" in provider.prompts[0]
    assert all("older" not in p for p in provider.prompts)  # the backlog is not replayed
    assert MarkStore(tmp_path).get(TIMELINE_UUID, kind="webhook_events") == E1


def test_event_after_a_mark_is_acted_on(platform, tmp_path):
    """Steady state: a delivery newer than the persisted event mark is acted on."""
    MarkStore(tmp_path).set(TIMELINE_UUID, E0, kind="webhook_events")
    serve_messages(platform, page())
    serve_events(platform, event_page(event(uuid=E1, payload="new"), event(uuid=E0, payload="old")))
    agent, provider = build_wake(tmp_path)

    posted = agent.wake()

    assert len(posted) == 1
    assert "new" in provider.prompts[0]
    assert "old" not in provider.prompts[0]  # the already-seen one is skipped
    assert MarkStore(tmp_path).get(TIMELINE_UUID, kind="webhook_events") == E1


def test_event_wake_is_idempotent_across_processes(platform, tmp_path):
    """A second process (same home) sees the persisted event mark and acts on nothing."""
    serve_messages(platform, page(), page())
    serve_events(
        platform,
        event_page(event(uuid=E0, payload="x")),
        event_page(event(uuid=E0, payload="x")),
    )
    first, _ = build_wake(tmp_path)
    assert len(first.wake(event_trigger=E0)) == 1

    second, second_provider = build_wake(tmp_path)
    assert second.wake(event_trigger=E0) == []  # E0 already handled
    assert second_provider.prompts == []  # the model was never consulted


def test_messages_and_events_both_surface_in_one_wake(platform, tmp_path):
    """One wake reconciles a new message and a new delivery, in the same session."""
    serve_messages(platform, page(message(uuid=M0, body="hi")))
    serve_events(platform, event_page(event(uuid=E0, payload="ping")))
    agent, provider = build_wake(tmp_path)

    posted = agent.wake(event_trigger=E0)

    assert len(posted) == 2  # a reply to the message and an action on the delivery
    assert any("[2026-06-04T00:00:00.000Z] john: hi" in p for p in provider.prompts)
    assert any("inbound webhook" in p for p in provider.prompts)
    # Each kind advanced its own mark, independently.
    assert MarkStore(tmp_path).get(TIMELINE_UUID) == M0
    assert MarkStore(tmp_path).get(TIMELINE_UUID, kind="webhook_events") == E0


def test_no_events_makes_no_provider_call(platform, tmp_path):
    """A wake with an event mark already at the newest delivery makes no model call."""
    MarkStore(tmp_path).set(TIMELINE_UUID, E0, kind="webhook_events")
    MarkStore(tmp_path).set(TIMELINE_UUID, M0)  # and no new messages either
    serve_messages(platform, page(message(uuid=M0, body="old")))
    serve_events(platform, event_page(event(uuid=E0, payload="old")))
    agent, provider = build_wake(tmp_path)

    assert agent.wake() == []
    assert provider.prompts == []


def test_trigger_past_the_fetch_window_is_fetched_not_dropped(platform, tmp_path):
    """A burst can push the triggering event past the bounded first-wake fetch; it must
    still be acted on (fetched directly), never silently replaced by the newest."""
    serve_messages(platform, page())
    # The window the first wake fetches does NOT contain E0 (the trigger) — a burst of
    # newer deliveries (E1) crowded it out (context_messages=1 forces a 1-event window).
    serve_events(platform, event_page(event(uuid=E1, payload="newer burst event")))
    # The trigger is reachable by uuid, the way the SDK fetches one event.
    platform.get(f"/webhook_events/{E0}").mock(
        return_value=httpx.Response(
            200, json={"webhook_event": event(uuid=E0, payload="THE TRIGGER")}
        )
    )
    agent, provider = build_wake(tmp_path, context_messages=1)

    posted = agent.wake(event_trigger=E0)

    # Both the named trigger (fetched directly) and the window's newer event are acted on.
    assert any("THE TRIGGER" in p for p in provider.prompts)
    assert any("newer burst event" in p for p in provider.prompts)
    assert len(posted) == 2
    assert (
        MarkStore(tmp_path).get(TIMELINE_UUID, kind="webhook_events") == E1
    )  # baselined to newest


def test_large_event_payload_is_truncated_with_a_pointer(platform, tmp_path):
    """A firehose payload is truncated, with a pointer to the webhook_events tool."""
    serve_messages(platform, page())
    big = "x" * (9 * 1024)
    serve_events(platform, event_page(event(uuid=E0, payload=big)))
    agent, provider = build_wake(tmp_path)

    agent.wake(event_trigger=E0)

    prompt = provider.prompts[0]
    assert "payload truncated" in prompt
    assert "webhook_events tool" in prompt
    assert len(prompt) < len(big)  # not the whole body


# --- newly-activated tasks (task.activated) ----------------------------------


def test_activated_task_is_carried_out(platform, tmp_path):
    """A task.activated wake makes the agent carry out the task's instructions."""
    serve_messages(platform, page())  # no messages
    serve_tasks(
        platform, task_page(task(uuid=T0, instructions="generate a silly monkey and post it"))
    )
    agent, provider = build_wake(tmp_path)

    posted = agent.wake()

    assert len(posted) == 1  # the agent acted on the activated task
    assert len(provider.prompts) == 1
    assert "generate a silly monkey and post it" in provider.prompts[0]
    assert "activated" in provider.prompts[0]
    assert T0 in SeenStore(tmp_path).all(TIMELINE_UUID, kind="tasks")


def test_only_activated_tasks_are_fetched(platform, tmp_path):
    """The reconcile asks the platform only for activated tasks, never pending ones."""
    serve_messages(platform, page())
    serve_tasks(platform, task_page(task(uuid=T0, instructions="do the thing")))
    agent, _ = build_wake(tmp_path)

    agent.wake()

    assert "status=activated" in str(platform.get("/tasks").calls.last.request.url)


def test_activated_task_is_idempotent_across_processes(platform, tmp_path):
    """A second process (same home) sees the seen-set and does not re-run the task."""
    serve_messages(platform, page(), page())
    serve_tasks(
        platform,
        task_page(task(uuid=T0, instructions="post the monkey")),
        task_page(task(uuid=T0, instructions="post the monkey")),
    )
    first, _ = build_wake(tmp_path)
    assert len(first.wake()) == 1

    second, second_provider = build_wake(tmp_path)
    assert second.wake() == []  # T0 already handled
    assert second_provider.prompts == []  # the model was never consulted


def test_activated_task_is_claimed_before_acting_so_a_crash_cannot_refire(platform, tmp_path):
    """THE BUG-1 REGRESSION: a task is recorded seen BEFORE its action runs, so an action
    that fails part-way — most importantly one that already posted a side effect whose
    `asset.created` re-wakes the agent — can never re-surface the still-`activated` task
    and re-run it. That act-then-record gap was the live monkey pile-up. At-most-once."""
    serve_messages(platform, page(), page())
    serve_tasks(
        platform,
        task_page(task(uuid=T0, instructions="generate a monkey and post it")),
        task_page(task(uuid=T0, instructions="generate a monkey and post it")),
    )

    class DiesOnTheTask:
        """Stands in for an action that blows up after the task was claimed."""

        prompts: list[str] = []

        def chat(self, messages, tools=None):
            raise RuntimeError("the action crashed mid-flight")

    first, _ = build_wake(tmp_path, DiesOnTheTask())
    with pytest.raises(RuntimeError):
        first.wake()
    # The crash propagated, but the task was already claimed — so it is recorded seen.
    assert T0 in SeenStore(tmp_path).all(TIMELINE_UUID, kind="tasks")

    # A later wake (working brain) does NOT re-run it: no re-fire, no monkey pile-up.
    second, second_provider = build_wake(tmp_path)
    assert second.wake() == []
    assert second_provider.prompts == []  # the model was never consulted again


def test_already_handled_task_is_skipped_when_a_new_one_activates(platform, tmp_path):
    """With T0 already done, a wake acts only on the newly-activated T1."""
    SeenStore(tmp_path).add(TIMELINE_UUID, T0, kind="tasks")
    serve_messages(platform, page())
    # Newest-first from the platform: T1 (new) then T0 (already handled).
    serve_tasks(
        platform,
        task_page(
            task(uuid=T1, instructions="newer task"),
            task(uuid=T0, instructions="older task"),
        ),
    )
    agent, provider = build_wake(tmp_path)

    posted = agent.wake()

    assert len(posted) == 1
    assert "newer task" in provider.prompts[0]
    assert "older task" not in provider.prompts[0]


def test_multiple_activated_tasks_done_oldest_first(platform, tmp_path):
    """A burst of unhandled activations is worked oldest-first, in schedule order."""
    serve_messages(platform, page())
    serve_tasks(
        platform,
        task_page(task(uuid=T1, instructions="second"), task(uuid=T0, instructions="first")),
    )
    agent, provider = build_wake(tmp_path)

    posted = agent.wake()

    assert len(posted) == 2
    assert provider.prompts[0].endswith("first")  # T0, the older task, first
    assert provider.prompts[1].endswith("second")


def test_a_large_activation_burst_is_fully_drained(platform, tmp_path):
    """All activated tasks are carried out, even a burst larger than the context cap —
    the task reconcile is not windowed, so none is silently dropped."""
    burst = 60  # larger than the default context_messages window (50)
    uuids = [f"019e7772-0000-7000-8000-{i:012d}" for i in range(burst)]
    # Newest-first from the platform (highest index first).
    tasks = [task(uuid=u, instructions=f"task {i}") for i, u in reversed(list(enumerate(uuids)))]
    serve_messages(platform, page())
    serve_tasks(platform, task_page(*tasks))
    agent, provider = build_wake(tmp_path)

    posted = agent.wake()

    assert len(posted) == burst  # every one acted on, not just the newest 50
    assert len(SeenStore(tmp_path).all(TIMELINE_UUID, kind="tasks")) == burst


def test_no_activated_tasks_makes_no_provider_call(platform, tmp_path):
    """A wake with nothing activated (and nothing else new) makes no model call."""
    MarkStore(tmp_path).set(TIMELINE_UUID, M0)  # no new messages
    serve_messages(platform, page(message(uuid=M0, body="old")))
    # tasks default to an empty page from the fixture
    agent, provider = build_wake(tmp_path)

    assert agent.wake() == []
    assert provider.prompts == []


# --- a peer's posted assets (+ the actor self-filter, the safety property) ---


def test_a_peer_posted_asset_is_surfaced(platform, tmp_path):
    """A file a peer shares is perceived on wake (the founder's minimum wake set)."""
    MarkStore(tmp_path).set(TIMELINE_UUID, A0, kind="assets")  # past the baseline
    serve_messages(platform, page())
    serve_assets(platform, asset_page(asset(uuid=A1, filename="diagram.png")))
    agent, provider = build_wake(tmp_path)

    posted = agent.wake()

    assert len(posted) == 1
    assert "john posted a file" in provider.prompts[0]
    assert "diagram.png" in provider.prompts[0]
    assert MarkStore(tmp_path).get(TIMELINE_UUID, kind="assets") == A1


def test_a_posted_image_is_shown_to_the_model(platform, tmp_path):
    """PERCEPTION: a peer's image is fetched and presented inline, so a vision-capable
    agent actually *sees* the picture on wake — not merely a description of it."""
    MarkStore(tmp_path).set(TIMELINE_UUID, A0, kind="assets")
    serve_messages(platform, page())
    serve_assets(platform, asset_page(asset(uuid=A1, filename="diagram.png")))
    agent, provider = build_wake(tmp_path)

    agent.wake()

    # The image rode into the model's input as a self-contained data URL (the same form
    # the `view` tool uses), captured at chat time before the session evicts the pixels.
    assert len(provider.last_images) == 1
    expected = "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode("ascii")
    assert provider.last_images[0].url == expected
    assert provider.last_images[0].alt == "diagram.png"


def test_a_shown_image_is_not_persisted_as_base64(platform, tmp_path):
    """COST DISCIPLINE: the presented pixels are evicted after the turn, so the data URL
    never lands in the on-disk transcript to be re-sent (and re-billed) on the next wake."""
    MarkStore(tmp_path).set(TIMELINE_UUID, A0, kind="assets")
    serve_messages(platform, page())
    serve_assets(platform, asset_page(asset(uuid=A1, filename="diagram.png")))
    agent, _ = build_wake(tmp_path)

    agent.wake()

    # The session transcript persists under `<home>/sessions/`; no base64 image survives
    # in it, but the text breadcrumb (the asset's filename) does.
    blob = "".join(p.read_text() for p in tmp_path.rglob("*.json"))
    assert "base64" not in blob
    assert "diagram.png" in blob


def test_a_non_image_asset_degrades_to_a_description(platform, tmp_path):
    """A file the wake can't show inline (here a PDF) is acknowledged by name and type,
    never an error — the graceful seam for media whose perception depth is out of scope."""
    MarkStore(tmp_path).set(TIMELINE_UUID, A0, kind="assets")
    serve_messages(platform, page())
    serve_assets(
        platform,
        asset_page(asset(uuid=A1, filename="report.pdf", content_type="application/pdf")),
    )
    agent, provider = build_wake(tmp_path)

    posted = agent.wake()

    assert len(posted) == 1  # acted on, gracefully
    assert provider.last_images == []  # nothing shown — a PDF isn't viewable
    assert "report.pdf" in provider.prompts[0]
    assert "application/pdf" in provider.prompts[0]  # the type is acknowledged


def test_an_image_whose_download_fails_degrades_gracefully(platform, tmp_path):
    """A fetch failure on a viewable image is not an error: the wake falls back to a
    description so the agent still perceives *that* a file was shared, just not its pixels."""
    MarkStore(tmp_path).set(TIMELINE_UUID, A0, kind="assets")
    serve_messages(platform, page())
    serve_assets(platform, asset_page(asset(uuid=A1, filename="broken.png")))
    # The blob fetch fails this wake (e.g. the blob store hiccuped).
    platform.get(path__regex=r"^/blobs/").mock(return_value=httpx.Response(503))
    agent, provider = build_wake(tmp_path)

    posted = agent.wake()

    assert len(posted) == 1  # still acted on, not crashed
    assert provider.last_images == []  # the picture couldn't be shown
    assert "broken.png" in provider.prompts[0]  # but the file is still acknowledged
    assert MarkStore(tmp_path).get(TIMELINE_UUID, kind="assets") == A1


def test_own_posted_asset_is_not_acted_on(platform, tmp_path):
    """THE SAFETY PROPERTY: the agent never acts on its own posted asset (e.g. an image
    it generated), so a generated image cannot wake it to generate another — no loop.
    The mark still advances, so the own asset is not re-scanned forever."""
    MarkStore(tmp_path).set(TIMELINE_UUID, A0, kind="assets")
    serve_messages(platform, page())
    # The newest asset is the agent's OWN — a generate_image output, say.
    serve_assets(platform, asset_page(asset(uuid=A1, filename="generated.png", mine=True)))
    agent, provider = build_wake(tmp_path)

    posted = agent.wake()

    assert posted == []  # nothing posted in response to its own file
    assert provider.prompts == []  # the model was never even consulted — no wake-loop
    assert MarkStore(tmp_path).get(TIMELINE_UUID, kind="assets") == A1  # but marked seen


def test_own_asset_is_skipped_but_a_peer_asset_beside_it_is_acted_on(platform, tmp_path):
    """The self-filter is per-item: the agent's own asset is skipped, a peer's is not,
    and the mark advances past both."""
    MarkStore(tmp_path).set(TIMELINE_UUID, A0, kind="assets")
    serve_messages(platform, page())
    # Newest-first: the agent's own A1, then John's... use A1 (mine) newest, plus a
    # peer asset that is also unseen.
    peer = asset(uuid="019e7781-9999-7ccc-8ddd-940516273849", filename="from-john.png")
    serve_assets(platform, asset_page(asset(uuid=A1, filename="mine.png", mine=True), peer))
    agent, provider = build_wake(tmp_path)

    posted = agent.wake()

    assert len(posted) == 1  # only the peer's asset drew a response
    assert "from-john.png" in provider.prompts[0]
    assert "mine.png" not in provider.prompts[0]
    assert MarkStore(tmp_path).get(TIMELINE_UUID, kind="assets") == A1  # mark past the newest


def test_first_asset_wake_without_a_trigger_acts_on_the_newest_asset(platform, tmp_path):
    """The router wakes with the timeline uuid alone (no `--asset`), so a peer's posted
    asset arrives with no trigger. The first wake acts on the newest unseen asset rather
    than baselining it away, so a peer's file actually surfaces under the real router
    contract — bounded to the newest, not a backlog replay."""
    serve_messages(platform, page())
    serve_assets(
        platform,
        asset_page(asset(uuid=A1, filename="newest.png"), asset(uuid=A0, filename="older.png")),
    )
    agent, provider = build_wake(tmp_path)

    posted = agent.wake()  # no asset_trigger — the real router contract

    assert len(posted) == 1
    assert "newest.png" in provider.prompts[0]
    assert all("older.png" not in p for p in provider.prompts)  # the backlog is not replayed
    assert MarkStore(tmp_path).get(TIMELINE_UUID, kind="assets") == A1


def test_first_asset_wake_on_own_post_is_self_filtered_without_a_trigger(platform, tmp_path):
    """The self-filter holds on the no-trigger first wake too: if the newest asset is the
    agent's own (a generated image), it is marked but not acted on — no wake-loop."""
    serve_messages(platform, page())
    serve_assets(platform, asset_page(asset(uuid=A1, filename="generated.png", mine=True)))
    agent, provider = build_wake(tmp_path)

    posted = agent.wake()

    assert posted == []  # own asset never acted on
    assert provider.prompts == []  # the model was never consulted — no loop
    assert MarkStore(tmp_path).get(TIMELINE_UUID, kind="assets") == A1  # but marked seen


def test_first_asset_wake_with_a_trigger_perceives_that_asset(platform, tmp_path):
    """On an asset.created wake the router names the asset; the first wake perceives it
    (rather than baselining it away), so the very first posted file is not missed."""
    serve_messages(platform, page())
    serve_assets(platform, asset_page(asset(uuid=A1, filename="just-posted.png")))
    agent, provider = build_wake(tmp_path)

    posted = agent.wake(asset_trigger=A1)

    assert len(posted) == 1
    assert "just-posted.png" in provider.prompts[0]
    assert MarkStore(tmp_path).get(TIMELINE_UUID, kind="assets") == A1


def test_own_asset_trigger_is_still_self_filtered(platform, tmp_path):
    """Even if a wake fires on the agent's own asset, the self-filter holds on the
    trigger path — no action, no loop."""
    serve_messages(platform, page())
    serve_assets(platform, asset_page(asset(uuid=A1, filename="mine.png", mine=True)))
    agent, provider = build_wake(tmp_path)

    posted = agent.wake(asset_trigger=A1)

    assert posted == []
    assert provider.prompts == []
    assert MarkStore(tmp_path).get(TIMELINE_UUID, kind="assets") == A1  # baselined, not acted


def test_asset_reconcile_is_idempotent_across_processes(platform, tmp_path):
    """A re-invoked wake sees the persisted asset mark and acts on nothing."""
    MarkStore(tmp_path).set(TIMELINE_UUID, A0, kind="assets")
    serve_messages(platform, page(), page())
    serve_assets(
        platform,
        asset_page(asset(uuid=A1, filename="x.png")),
        asset_page(asset(uuid=A1, filename="x.png")),
    )
    first, _ = build_wake(tmp_path)
    assert len(first.wake()) == 1

    second, second_provider = build_wake(tmp_path)
    assert second.wake() == []
    assert second_provider.prompts == []


# --- the session key + shared memory -----------------------------------------


def test_wake_uses_a_timeline_scoped_session(platform, tmp_path):
    """Each wake runs the `timeline:<uuid>` session — one identity, channel-keyed."""
    serve_messages(platform, page(message(uuid=M0, body="hi")))
    agent, _ = build_wake(tmp_path)
    agent.wake()

    assert agent.source == f"timeline:{TIMELINE_UUID}"
    assert f"timeline:{TIMELINE_UUID}" in agent.harness.sessions


def test_wake_binds_platform_tools_to_its_timeline(platform, tmp_path):
    """A platform-aware tool in the harness is wired to this wake's client + timeline."""
    from basecradle_harness import AssetsTool

    client = BaseCradle(token=FAKE_TOKEN)
    harness = Harness(CountingProvider(), tools=[AssetsTool()], home=tmp_path)
    WakeAgent(harness, timeline=TIMELINE_UUID, client=client, onboard=False)

    assets = harness.tools.get("assets")
    assert assets.bound is True
    assert assets.context.timeline == TIMELINE_UUID
    assert assets.context.client is client
    assert assets.context.home == tmp_path


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
    monkeypatch.setenv("AI_MODEL", "gpt-4o")
    monkeypatch.setenv("AI_API_KEY", "sk-test-key")
    monkeypatch.setenv("HARNESS_HOME", str(tmp_path))
    monkeypatch.setenv("HARNESS_ONBOARD", "0")
    monkeypatch.delenv("BASECRADLE_TIMELINE", raising=False)
    monkeypatch.delenv("BASECRADLE_MESSAGE", raising=False)
    monkeypatch.delenv("BASECRADLE_EVENT", raising=False)
    return tmp_path


def _serve_openai_and_messages(platform, *pages):
    """A plain-text model reply plus the message pages — enough for `main` to run live.

    Wake mode runs the default @jt stack — the ``openai`` SDK on the **Responses** surface — so
    the model call lands on ``/responses`` (SDK-validated body), not ``/chat/completions``.
    """
    platform.post("https://api.openai.com/v1/responses").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "resp-wake",
                "object": "response",
                "created_at": 0,
                "model": "gpt-4o",
                "output": [
                    {
                        "id": "msg-wake",
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "On it.", "annotations": []}],
                    }
                ],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
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


def test_main_acts_on_a_webhook_event_trigger(platform, wake_env):
    """`--event` forwards the triggering delivery through to the wake (a manual path)."""
    _serve_openai_and_messages(platform, page())  # no messages
    serve_events(platform, event_page(event(uuid=E0, payload='{"x":1}')))

    assert main(["--timeline", TIMELINE_UUID, "--event", E0]) == 0
    assert platform.post(f"/timelines/{TIMELINE_UUID}/messages").called  # it acted on the delivery


def test_main_acts_on_a_webhook_delivery_with_no_event_flag(platform, wake_env):
    """THE REAL ROUTER INVOCATION: `--timeline <uuid>` alone, no `--event`. A delivery
    present on the timeline is still acted on — the bug-2 fix at the CLI boundary."""
    _serve_openai_and_messages(platform, page())  # no messages
    serve_events(platform, event_page(event(uuid=E0, payload="PAPAYA-CLEAN-42")))

    assert main(["--timeline", TIMELINE_UUID]) == 0  # exactly what the router runs
    assert platform.post(f"/timelines/{TIMELINE_UUID}/messages").called  # acted, unaided


def test_main_without_a_timeline_errors(wake_env):
    """No --timeline and no BASECRADLE_TIMELINE → argparse exits non-zero."""
    with pytest.raises(SystemExit) as exit_info:
        main([])
    assert exit_info.value.code != 0


def test_main_version_flag_prints_harness_and_sdk_versions_and_exits_zero(capsys):
    """`--version` is the fleet drift-guard's cheap probe: print the installed harness **and**
    vendor-SDK versions (an upgrade tracks both — issue #158), exit 0, touch no timeline, model,
    or credential."""
    import openai

    from basecradle_harness import __version__

    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    out = capsys.readouterr().out.strip()
    assert out == f"basecradle-harness-wake {__version__} · openai SDK {openai.__version__}"


def test_main_resolved_config_prints_ground_truth_json_and_exits_zero(wake_env, capsys):
    """`--resolved-config` is the NOC's ground-truth deploy probe (issue #174): print the live,
    *resolved* config + active tool set as stable JSON, exit 0, and need no timeline, model call,
    or platform network (no `platform` respx mock here — it must not hit the API)."""
    from basecradle_harness import __version__

    assert main(["--resolved-config"]) == 0  # no --timeline; introspection is timeline-free
    report = json.loads(capsys.readouterr().out)

    # The additive contract: every documented field is present and is ground truth, not a declaration.
    assert report["harness_version"] == __version__
    assert report["ai_provider"] == "openai"
    assert report["ai_sdk"] == "openai"
    assert report["ai_sdk_surface"] == "responses"  # the openai adapter's default surface
    assert report["ai_model"] == "gpt-4o"  # the wake_env AI_MODEL, reported verbatim
    import openai

    assert report["ai_sdk_version"] == openai.__version__
    # The resolved active tool set — the platform tools that actually activate under the locked
    # policy, not a shipped-default list. The benign platform reads/writes are always present.
    assert {"memory", "users", "messages", "timelines", "lock", "delete"} <= set(report["tools"])
    # Powerful tools are opt-in (issue #168), so none activate in a default config.
    assert "generate_image" not in report["tools"]
    assert "web_search" not in report["builtins"]
    # And the opt-in manifest (issue #181) is empty for that safe default — nothing opted in.
    assert report["opt_in_tools"] == []


def test_resolved_config_is_side_effect_free_without_a_model_or_key(wake_env, monkeypatch):
    """It resolves through the live code paths but builds **no** model provider, so it works with
    no `AI_API_KEY` and reports an unset `AI_MODEL` as `None` rather than raising the provider
    build's "AI_MODEL is required" — safe to run repeatedly over SSH against a live agent home."""
    monkeypatch.delenv("AI_API_KEY", raising=False)
    monkeypatch.delenv("AI_MODEL", raising=False)

    report = resolved_config()

    assert report["ai_model"] is None  # unset → None, never a raise
    assert report["tools"]  # the tool set still resolves with no key / no model


def test_resolved_config_reports_active_opt_in_stems(wake_env, monkeypatch, tmp_path):
    """The opt-in manifest (issue #181) the NOC's fleet-drift audit keys on: an agent with
    powerful tools opted into its overlay reports their source-file **stems** (the inventory
    key), like-for-like — even when a stem fans out to several names (``code_execution``)."""
    cfg = tmp_path / "cfg"
    monkeypatch.setenv("BASECRADLE_CONFIG_HOME", str(cfg))
    install(cfg, provider="openai", opt_in=["generate_image", "code_execution"])

    report = resolved_config()

    # Stems, not resolved names: code_execution fans out (the code_interpreter built-in + the
    # code_attach tool) yet lists once; generate_image's stem matches its name.
    assert report["opt_in_tools"] == ["code_execution", "generate_image"]
    # Cross-check the fan-out is real in the resolved active set (built-in + tool both present).
    assert "code_interpreter" in report["builtins"]
    assert "code_attach" in report["tools"]


def test_main_resolved_config_exits_nonzero_on_a_misconfigured_provider(
    wake_env, monkeypatch, capsys
):
    """A resolution error (an unknown AI_PROVIDER) is the verifier's honest "misconfigured" signal:
    a clean non-zero exit with the reason on stderr, never a raw traceback."""
    monkeypatch.setenv("AI_PROVIDER", "bogus")

    assert main(["--resolved-config"]) == 1
    assert "Unknown AI_PROVIDER 'bogus'" in capsys.readouterr().err


def test_main_returns_nonzero_when_home_is_missing(platform, wake_env, monkeypatch):
    """A hard config failure (no HARNESS_HOME) exits non-zero so the router reports it."""
    monkeypatch.delenv("HARNESS_HOME", raising=False)

    assert main(["--timeline", TIMELINE_UUID]) == 1


def test_main_returns_nonzero_on_missing_provider_config(platform, wake_env, monkeypatch):
    monkeypatch.delenv("AI_MODEL", raising=False)

    assert main(["--timeline", TIMELINE_UUID]) == 1


# --- NOC synthetic-probe short-circuit (issue #106) --------------------------
#
# A woken harness recognizes a signed NOC probe (see `_probe`) and acks it at the
# reconcile layer WITHOUT a model call, so the message-seam heartbeat is token-free at
# rest. These pin the wake-level contract: ack the probe, no provider call, advance the
# mark, leave the transcript clean; an unsigned/forged marker falls through to the model.

PROBE_SECRET = "noc-probe-secret-do-not-use-in-prod"
PROBE_NONCE = "0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d"


def probe_marker(nonce=PROBE_NONCE, secret=PROBE_SECRET):
    """A correctly-signed BCNOC1 marker line (mirrors basecradle-noc's marker.mint)."""
    sig = hmac.new(secret.encode(), f"BCNOC1 {nonce}".encode(), sha256).hexdigest()
    return f"BCNOC1 {nonce} {sig}"


def _conversational_turns(agent):
    """The user/assistant turns in the wake's session — empty means a clean transcript."""
    history = agent.harness.session(agent.source).history
    return [turn for turn in history if turn.role in ("user", "assistant")]


def test_a_signed_probe_is_acked_without_the_model(platform, tmp_path):
    """A valid probe from a peer → a BCNOC1-ACK, zero model calls, mark advanced, clean transcript."""
    body = f"NOC message-seam probe — please disregard.\n{probe_marker()}"
    serve_messages(platform, page(message(uuid=M0, body=body)))
    agent, provider = build_wake(tmp_path, probe_secret=PROBE_SECRET)

    posted = agent.wake()

    assert len(posted) == 1  # the ack
    assert provider.prompts == []  # the model never ran — token-free
    sent = platform.post(f"/timelines/{TIMELINE_UUID}/messages").calls.last.request
    assert json.loads(sent.content) == {"message": {"body": f"BCNOC1-ACK {PROBE_NONCE}"}}
    assert agent.marks.get(TIMELINE_UUID) == M0  # mark advanced exactly as a normal reply
    assert _conversational_turns(agent) == []  # probe never entered the session transcript


def test_an_acked_probe_is_idempotent_across_wakes(platform, tmp_path):
    """The mark advance means a second process re-reconciling acks nothing more."""
    body = probe_marker()
    serve_messages(
        platform,
        page(message(uuid=M0, body=body)),  # first wake sees the probe
        page(message(uuid=M0, body=body)),  # second wake re-reads, now past the mark
    )
    agent1, _ = build_wake(tmp_path, probe_secret=PROBE_SECRET)
    agent1.wake()

    agent2, provider2 = build_wake(tmp_path, probe_secret=PROBE_SECRET)
    posted = agent2.wake()

    assert posted == []  # nothing newer than the mark
    assert provider2.prompts == []


def test_a_forged_marker_falls_through_to_the_model(platform, tmp_path):
    """Right shape, wrong signature → an ordinary message answered by the model."""
    forged = f"BCNOC1 {PROBE_NONCE} {'0' * 64}"
    serve_messages(platform, page(message(uuid=M0, body=forged)))
    agent, provider = build_wake(tmp_path, probe_secret=PROBE_SECRET)

    posted = agent.wake()

    assert len(posted) == 1
    assert provider.prompts == [
        f"[2026-06-04T00:00:00.000Z] john: {forged}"
    ]  # the model ran on it as a normal message
    sent = platform.post(f"/timelines/{TIMELINE_UUID}/messages").calls.last.request
    assert json.loads(sent.content) == {"message": {"body": "Hello, John."}}


def test_without_a_probe_secret_a_valid_marker_is_an_ordinary_message(platform, tmp_path):
    """Feature off (no NOC_PROBE_SECRET) → even a real probe goes to the model, unchanged."""
    serve_messages(platform, page(message(uuid=M0, body=probe_marker())))
    agent, provider = build_wake(tmp_path)  # no probe_secret

    agent.wake()

    assert provider.prompts == [
        f"[2026-06-04T00:00:00.000Z] john: {probe_marker()}"
    ]  # short-circuit disabled


def test_a_self_authored_probe_is_filtered_not_acked(platform, tmp_path):
    """Self-filter precedence: a probe the agent itself posted is skipped, never acked."""
    serve_messages(platform, page(message(uuid=M0, body=probe_marker(), mine=True)))
    agent, provider = build_wake(tmp_path, probe_secret=PROBE_SECRET)

    posted = agent.wake()

    assert posted == []  # own item: never acted on
    assert provider.prompts == []
    assert agent.marks.get(TIMELINE_UUID) == M0  # but the mark still advances


def test_a_blank_probe_secret_keeps_the_short_circuit_off(platform, tmp_path):
    """A set-but-empty secret must not enable verification against an empty HMAC key."""
    serve_messages(platform, page(message(uuid=M0, body=probe_marker())))
    agent, provider = build_wake(tmp_path, probe_secret="")  # blank, e.g. an unfilled env var

    agent.wake()

    assert provider.prompts == [
        f"[2026-06-04T00:00:00.000Z] john: {probe_marker()}"
    ]  # treated as an ordinary message


# --- NOC probe short-circuit: the TASK seam (issue #110) ---------------------
#
# The same recognize-and-ack discipline as the message seam, but the marker rides the
# task's *instructions*. The load-bearing subtlety is ordering: `_act_on` checks `probe`
# BEFORE the atomic claim, so a probe task is acked at-least-once (post, then record) and
# never claimed — the safe failure direction for a monitor (task-seam.md §4).


def test_a_signed_probe_task_is_acked_without_the_model(platform, tmp_path):
    """A probe in a task's instructions → a BCNOC1-ACK, zero model calls, task recorded
    seen, clean transcript. The task-seam heartbeat runs token-free at rest."""
    instructions = f"NOC task-seam probe — please disregard.\n{probe_marker()}"
    serve_messages(platform, page())
    serve_tasks(platform, task_page(task(uuid=T0, instructions=instructions)))
    agent, provider = build_wake(tmp_path, probe_secret=PROBE_SECRET)

    posted = agent.wake()

    assert len(posted) == 1  # the ack
    assert provider.prompts == []  # the model never ran — token-free
    sent = platform.post(f"/timelines/{TIMELINE_UUID}/messages").calls.last.request
    assert json.loads(sent.content) == {"message": {"body": f"BCNOC1-ACK {PROBE_NONCE}"}}
    assert T0 in SeenStore(tmp_path).all(TIMELINE_UUID, kind="tasks")  # recorded seen
    assert _conversational_turns(agent) == []  # probe never entered the session transcript


def test_a_probe_task_is_acked_at_least_once_not_claimed(platform, tmp_path):
    """LOAD-BEARING: `_act_on` checks `probe` before claiming, so a probe task is acked
    at-least-once (post, THEN record), never pre-claimed. If the ack post is refused, the
    task is NOT recorded seen — it retries next wake. This is the safe failure direction:
    were the probe path 'fixed' to claim-first, the task would be marked seen with no ack
    ever posted → the loop never closes → a false FAIL (task-seam.md §4).

    Under B2 a refused post no longer crashes the wake; the at-least-once guarantee is
    preserved by recording only after a *successful* ack, and the probe seam stays
    trace-free — a refused ack degrades SILENTLY (no transcript note), never polluting the
    deliberately-empty probe transcript or mislabeling the heartbeat ack as a 'reply'."""
    serve_messages(platform, page())
    serve_tasks(platform, task_page(task(uuid=T0, instructions=probe_marker())))
    # The ack post is refused this wake (a locked timeline / transient 5xx).
    platform.post(f"/timelines/{TIMELINE_UUID}/messages").mock(
        return_value=httpx.Response(500, json={"error": "boom"})
    )
    agent, provider = build_wake(tmp_path, probe_secret=PROBE_SECRET)

    posted = agent.wake()  # B2: degrades gracefully, never crashes

    assert posted == []  # the ack never made it out
    assert provider.prompts == []  # still no model call — the probe was recognized
    # NOT recorded: a failed ack must retry, never mark the task seen with no ack posted.
    assert T0 not in SeenStore(tmp_path).all(TIMELINE_UUID, kind="tasks")
    # Trace-free: a refused probe ack writes NO note, so the probe transcript stays empty.
    assert agent.harness.session(agent.source).history == []


def test_a_task_without_a_marker_falls_through_to_the_model(platform, tmp_path):
    """An ordinary activated task (no marker) is still carried out by the model, unchanged."""
    serve_messages(platform, page())
    serve_tasks(platform, task_page(task(uuid=T0, instructions="post the daily summary")))
    agent, provider = build_wake(tmp_path, probe_secret=PROBE_SECRET)

    posted = agent.wake()

    assert len(posted) == 1
    assert len(provider.prompts) == 1
    assert "post the daily summary" in provider.prompts[0]  # the model ran on it


# --- NOC probe short-circuit: the WEBHOOK seam (issue #110) ------------------
#
# The closest of the three to the message seam: a probe webhook delivery is acked
# at-least-once and never claimed; the marker rides the event *payload*.
# The short-circuit runs inside `_act_on`, after `_bootstrap_stream` selects the item, so
# the #100 cold-first-wake bootstrap (newest unseen delivery only) is preserved.


def test_a_signed_probe_event_is_acked_without_the_model(platform, tmp_path):
    """A probe in a webhook payload, on a cold first wake → a BCNOC1-ACK, zero model
    calls, event mark advanced, clean transcript. The webhook-seam heartbeat is token-free
    at rest. This is the cold-first-wake path (no mark, no trigger → newest delivery), so
    it also pins that the #100 bootstrap composes with the short-circuit: one delivery on a
    quiet probe timeline, so newest = the probe, and it is acked rather than sent to the model."""
    serve_messages(platform, page())
    serve_events(platform, event_page(event(uuid=E0, payload=f"webhook probe\n{probe_marker()}")))
    agent, provider = build_wake(tmp_path, probe_secret=PROBE_SECRET)

    posted = agent.wake()

    assert len(posted) == 1  # the ack
    assert provider.prompts == []  # the model never ran — token-free
    sent = platform.post(f"/timelines/{TIMELINE_UUID}/messages").calls.last.request
    assert json.loads(sent.content) == {"message": {"body": f"BCNOC1-ACK {PROBE_NONCE}"}}
    assert MarkStore(tmp_path).get(TIMELINE_UUID, kind="webhook_events") == E0  # mark advanced
    assert _conversational_turns(agent) == []  # probe never entered the session transcript


def test_a_probe_event_after_a_mark_is_acked(platform, tmp_path):
    """Steady state (mark exists): a probe delivery newer than the mark is acked token-free,
    and the older already-seen delivery is left alone."""
    MarkStore(tmp_path).set(TIMELINE_UUID, E0, kind="webhook_events")
    serve_messages(platform, page())
    serve_events(
        platform,
        event_page(event(uuid=E1, payload=probe_marker()), event(uuid=E0, payload="old")),
    )
    agent, provider = build_wake(tmp_path, probe_secret=PROBE_SECRET)

    posted = agent.wake()

    assert len(posted) == 1
    assert provider.prompts == []  # no model call
    assert MarkStore(tmp_path).get(TIMELINE_UUID, kind="webhook_events") == E1


def test_an_event_without_a_marker_falls_through_to_the_model(platform, tmp_path):
    """An ordinary inbound delivery (no marker) still reaches the model, unchanged."""
    serve_messages(platform, page())
    serve_events(platform, event_page(event(uuid=E0, payload='{"action":"opened"}')))
    agent, provider = build_wake(tmp_path, probe_secret=PROBE_SECRET)

    posted = agent.wake()

    assert len(posted) == 1
    assert len(provider.prompts) == 1
    assert "inbound webhook" in provider.prompts[0]  # the model ran on it


# --- NOC probe short-circuit: the ASSET seam (issue #114) --------------------
#
# The 4th seam. The marker rides the asset's **description** — the asset analog of a
# message body / task instructions / webhook payload (see `_asset_marker_carrier`). Like
# the message and webhook seams the probe is at-least-once and never claimed, and it is
# acked *before* the asset's file is ever fetched, so a synthetic asset probe costs no
# download and no model call. The carrier field (`description`) is the contract the NOC's
# asset probe must agree with.


def test_a_signed_probe_asset_is_acked_without_the_model(platform, tmp_path):
    """A probe in an asset's description → a BCNOC1-ACK, zero model calls, mark advanced,
    clean transcript. The asset-seam heartbeat runs token-free at rest."""
    MarkStore(tmp_path).set(TIMELINE_UUID, A0, kind="assets")
    serve_messages(platform, page())
    serve_assets(
        platform,
        asset_page(asset(uuid=A1, filename="probe.png", description=probe_marker())),
    )
    agent, provider = build_wake(tmp_path, probe_secret=PROBE_SECRET)

    posted = agent.wake()

    assert len(posted) == 1  # the ack
    assert provider.prompts == []  # the model never ran — token-free
    sent = platform.post(f"/timelines/{TIMELINE_UUID}/messages").calls.last.request
    assert json.loads(sent.content) == {"message": {"body": f"BCNOC1-ACK {PROBE_NONCE}"}}
    assert MarkStore(tmp_path).get(TIMELINE_UUID, kind="assets") == A1  # mark advanced
    assert _conversational_turns(agent) == []  # probe never entered the session transcript


def test_a_probe_asset_is_acked_before_its_file_is_fetched(platform, tmp_path):
    """The short-circuit runs before perception: a probe asset is acked with no blob
    download at all — the asset analog of acking before a model call."""
    MarkStore(tmp_path).set(TIMELINE_UUID, A0, kind="assets")
    serve_messages(platform, page())
    serve_assets(
        platform,
        asset_page(asset(uuid=A1, filename="probe.png", description=probe_marker())),
    )
    agent, _ = build_wake(tmp_path, probe_secret=PROBE_SECRET)

    agent.wake()

    assert not platform.get(path__regex=r"^/blobs/").called  # the file was never fetched


def test_a_probe_asset_on_a_cold_first_wake_is_acked(platform, tmp_path):
    """The #100 cold-first-wake bootstrap (no mark, no trigger → newest unseen asset)
    composes with the short-circuit: on a quiet probe timeline the newest asset is the
    probe, and it is acked token-free rather than perceived."""
    serve_messages(platform, page())
    serve_assets(
        platform, asset_page(asset(uuid=A1, filename="probe.png", description=probe_marker()))
    )
    agent, provider = build_wake(tmp_path, probe_secret=PROBE_SECRET)

    posted = agent.wake()  # no asset_trigger — the real router contract

    assert len(posted) == 1
    assert provider.prompts == []  # the model never ran
    sent = platform.post(f"/timelines/{TIMELINE_UUID}/messages").calls.last.request
    assert json.loads(sent.content) == {"message": {"body": f"BCNOC1-ACK {PROBE_NONCE}"}}
    assert MarkStore(tmp_path).get(TIMELINE_UUID, kind="assets") == A1


def test_an_asset_without_a_marker_is_perceived_not_acked(platform, tmp_path):
    """An ordinary posted file (no marker in its description) is perceived by the model,
    unchanged — the short-circuit only fires on a correctly-signed probe."""
    MarkStore(tmp_path).set(TIMELINE_UUID, A0, kind="assets")
    serve_messages(platform, page())
    serve_assets(platform, asset_page(asset(uuid=A1, filename="diagram.png")))
    agent, provider = build_wake(tmp_path, probe_secret=PROBE_SECRET)

    posted = agent.wake()

    assert len(posted) == 1
    assert len(provider.prompts) == 1
    assert "diagram.png" in provider.prompts[0]  # the model perceived it


def test_a_self_authored_probe_asset_is_filtered_not_acked(platform, tmp_path):
    """Self-filter precedence holds on the asset seam too: a probe asset the agent itself
    posted is skipped, never acked (and its file is never fetched)."""
    MarkStore(tmp_path).set(TIMELINE_UUID, A0, kind="assets")
    serve_messages(platform, page())
    serve_assets(
        platform,
        asset_page(asset(uuid=A1, filename="mine.png", description=probe_marker(), mine=True)),
    )
    agent, provider = build_wake(tmp_path, probe_secret=PROBE_SECRET)

    posted = agent.wake()

    assert posted == []  # own item: never acted on, never acked
    assert provider.prompts == []
    assert MarkStore(tmp_path).get(TIMELINE_UUID, kind="assets") == A1  # but the mark advances


# --- the persistent Turn-0 brief (Phase 2 · Group 3) -------------------------


def test_the_brief_is_reasserted_on_every_wake(platform, tmp_path):
    """Turn 0 is persistent: the brief lands again each wake, not just at turn 1.

    Two wakes over one home (two router-spawned processes). Each engages the model, so each
    injects a fresh brief — the transcript carries it twice, recent in the conversation
    rather than aging out at the top.
    """
    serve_dashboard_md(platform)
    manifest = [("memory", None), ("lock", "one-way and irreversible.")]

    serve_messages(platform, page(message(uuid=M0, body="What's the status?")))
    agent1, _ = build_wake(tmp_path, onboard=True, tool_manifest=manifest)
    agent1.wake()
    assert len(_brief_turns(agent1)) == 1

    # A fresh process: the mark is now M0, so wake 2 replies only to the newer M1.
    serve_messages(
        platform, page(message(uuid=M1, body="any update?"), message(uuid=M0, body="hi"))
    )
    agent2, _ = build_wake(tmp_path, onboard=True, tool_manifest=manifest)
    agent2.wake()

    assert len(_brief_turns(agent2)) == 2  # re-asserted, persisted across the two processes


def test_the_brief_composes_all_four_parts(platform, tmp_path):
    """The brief is initialize.md + the live tool manifest + the live dashboard + personality."""
    serve_dashboard_md(platform, text="# Live Dashboard\n\nWho you are, where everything is.\n")
    serve_messages(platform, page(message(uuid=M0, body="hi")))
    agent, _ = build_wake(
        tmp_path,
        onboard=True,
        tool_manifest=[("memory", None), ("lock", "one-way and irreversible.")],
    )

    agent.wake()

    brief = _brief_turns(agent)[0].content
    assert "How to operate here" in brief  # 1. initialize.md (provider-independent guidance)
    assert "Trust is directional in storage, mutual at the gate." in brief  # the B6 trust note
    assert "Your active tools right now:" in brief  # 2. generated manifest…
    assert "- lock — one-way and irreversible." in brief  # …with the optional per-tool note
    assert "# Live Dashboard" in brief  # 3. the live dashboard.md primer
    assert "You are a helpful peer on BaseCradle." in brief  # 4. the packaged personality


def test_a_dashboard_fetch_failure_does_not_break_the_wake(platform, tmp_path):
    """A failed dashboard fetch degrades gracefully — the brief is composed from the rest."""
    platform.get("/users/dashboard.md").mock(return_value=httpx.Response(503))
    serve_messages(platform, page(message(uuid=M0, body="hi")))
    agent, _ = build_wake(tmp_path, onboard=True, tool_manifest=[("memory", None)])

    posted = agent.wake()

    assert len(posted) == 1  # the wake still replied — the fetch failure never broke it
    brief = _brief_turns(agent)[0].content
    assert "How to operate here" in brief  # initialize.md present…
    assert "Your active tools right now:" in brief  # …and the manifest…
    assert "You are a helpful peer on BaseCradle." in brief  # …and the personality, sans dashboard


def test_onboarding_off_asserts_no_brief(platform, tmp_path):
    """`onboard=False` wakes with only the operator's charter — no persistent brief."""
    serve_messages(platform, page(message(uuid=M0, body="hi")))
    agent, _ = build_wake(tmp_path, onboard=False)

    agent.wake()

    assert _brief_turns(agent) == []


def test_an_idle_wake_asserts_no_brief_and_skips_the_dashboard_fetch(platform, tmp_path):
    """Nothing unseen → the model is never engaged → no brief, and no live dashboard fetch.

    The lazy, once-per-wake assertion means an idle (or probe-only) wake pays nothing: it
    neither bloats the transcript with a brief nor fetches the live dashboard.
    """
    route = serve_dashboard_md(platform)
    MarkStore(tmp_path).set(TIMELINE_UUID, M0)  # caught up → nothing new this wake
    serve_messages(platform, page(message(uuid=M0, body="hi")))
    agent, provider = build_wake(tmp_path, onboard=True, tool_manifest=[("memory", None)])

    posted = agent.wake()

    assert posted == []
    assert provider.prompts == []  # the model was never engaged
    assert _brief_turns(agent) == []
    assert not route.called  # lazy: no engagement → the live dashboard was never fetched


def test_the_brief_precedes_the_item_it_governs(platform, tmp_path):
    """The brief lands just *ahead* of the user turn it should govern — order is load-bearing."""
    serve_dashboard_md(platform)
    serve_messages(platform, page(message(uuid=M0, body="What's the status?")))
    agent, _ = build_wake(tmp_path, onboard=True, tool_manifest=[("memory", None)])

    agent.wake()

    history = agent.harness.session(agent.source).history
    roles = [m.role for m in history]
    brief_idx = next(i for i, m in enumerate(history) if m in _brief_turns(agent))
    user_idx = next(i for i, m in enumerate(history) if m.role == "user")
    assert brief_idx < user_idx  # brief first, then the message it contextualizes
    assert "assistant" in roles  # and the reply followed


def test_a_brief_composition_failure_does_not_break_the_wake(platform, tmp_path, monkeypatch):
    """A raise inside brief composition (e.g. an IO error reading a prompt file) degrades to
    no brief — the wake still replies, never crashes. Same invariant the dashboard fetch holds."""
    import basecradle_harness._wake as wake_mod

    def boom(*args, **kwargs):
        raise OSError("permission denied reading prompts/initialize.md")

    monkeypatch.setattr(wake_mod, "prompt_text", boom)
    serve_dashboard_md(platform)
    serve_messages(platform, page(message(uuid=M0, body="hi")))
    agent, _ = build_wake(tmp_path, onboard=True, tool_manifest=[("memory", None)])

    posted = agent.wake()

    assert len(posted) == 1  # the wake still replied despite the compose failure
    assert _brief_turns(agent) == []  # …with no brief, rather than crashing


# --- Group 6: the cross-wake circuit-breaker ---------------------------------


class FakeClock:
    """A deterministic, advanceable clock — so a synthetic wake burst is reproducible."""

    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def _alert_bodies(platform):
    """Every message body posted to the timeline — replies and breaker alerts alike."""
    calls = platform.post(f"/timelines/{TIMELINE_UUID}/messages").calls
    return [json.loads(c.request.content)["message"]["body"] for c in calls]


def test_wake_breaker_trips_over_the_cap_and_auto_resets(tmp_path):
    """The breaker state machine, unit-level: under the cap is fine, over it trips once, a
    continuing burst stays tripped, and once the window clears past the cooldown it auto-resets."""
    clock = FakeClock()
    breaker = WakeBreaker(tmp_path, max_wakes=3, window=60.0, now=clock)

    # Three wakes at the same instant: at the cap, not over it — no trip.
    for i in range(3):
        decision = breaker.record_and_check(TIMELINE_UUID)
        assert decision == BreakerDecision(
            short_circuit=False, tripped=False, reset=False, count=i + 1
        )

    # The fourth wake is over the cap → TRIP (the one-time transition).
    decision = breaker.record_and_check(TIMELINE_UUID)
    assert decision.short_circuit and decision.tripped and decision.count == 4
    assert breaker.tripped(TIMELINE_UUID)

    # A fifth wake while the burst continues: still short-circuits, but it is *not* a fresh
    # trip transition (so the caller won't re-alert).
    decision = breaker.record_and_check(TIMELINE_UUID)
    assert decision.short_circuit and not decision.tripped and not decision.reset

    # The burst stops; advance past the window + cooldown so it clears → AUTO-RESET.
    clock.advance(200)
    decision = breaker.record_and_check(TIMELINE_UUID)
    assert decision.reset and not decision.short_circuit
    assert not breaker.tripped(TIMELINE_UUID)


def test_wake_breaker_normal_load_never_trips(tmp_path):
    """A steady, human-paced cadence (one wake every 30 s, cap 10/60 s) never trips —
    legitimate multi-peer activity must stay clear of the breaker."""
    clock = FakeClock()
    breaker = WakeBreaker(tmp_path, max_wakes=10, window=60.0, now=clock)
    for _ in range(50):
        clock.advance(30)  # at most ~2 wakes in any 60 s window
        decision = breaker.record_and_check(TIMELINE_UUID)
        assert not decision.short_circuit and not decision.tripped


def test_wake_breaker_disabled_when_cap_is_zero(tmp_path):
    """Cap 0 is the operator escape hatch: the breaker is off and never short-circuits."""
    breaker = WakeBreaker(tmp_path, max_wakes=0, window=60.0)
    assert not breaker.enabled
    for _ in range(100):
        assert not breaker.record_and_check(TIMELINE_UUID).short_circuit
    assert not breaker.tripped(TIMELINE_UUID)


def test_wake_breaker_trip_marker_persists_across_processes(tmp_path):
    """The trip marker is durable: a brand-new breaker (a fresh process) over the same home
    sees the timeline is tripped and keeps short-circuiting — wake mode is process-per-event."""
    clock = FakeClock()
    first = WakeBreaker(tmp_path, max_wakes=1, window=60.0, now=clock)
    first.record_and_check(TIMELINE_UUID)  # count 1 — at the cap
    assert first.record_and_check(TIMELINE_UUID).tripped  # count 2 — over → trip

    second = WakeBreaker(tmp_path, max_wakes=1, window=60.0, now=clock)
    assert second.tripped(TIMELINE_UUID)
    assert second.record_and_check(TIMELINE_UUID).short_circuit


def test_wake_breaker_from_env_reads_tunables(tmp_path, monkeypatch):
    """The three env knobs configure the breaker; cooldown defaults to the window when unset."""
    monkeypatch.setenv("HARNESS_WAKE_BREAKER_MAX", "5")
    monkeypatch.setenv("HARNESS_WAKE_BREAKER_WINDOW", "30")
    monkeypatch.setenv("HARNESS_WAKE_BREAKER_COOLDOWN", "45")
    breaker = WakeBreaker.from_env(tmp_path)
    assert (breaker.max_wakes, breaker.window, breaker.cooldown) == (5, 30.0, 45.0)

    monkeypatch.delenv("HARNESS_WAKE_BREAKER_COOLDOWN")
    assert WakeBreaker.from_env(tmp_path).cooldown == 30.0  # defaults to the window

    for var in ("HARNESS_WAKE_BREAKER_MAX", "HARNESS_WAKE_BREAKER_WINDOW"):
        monkeypatch.delenv(var)
    default = WakeBreaker.from_env(tmp_path)
    assert (default.max_wakes, default.window) == (10, 60.0)  # generous safe defaults


def _wake_with_breaker(tmp_path, provider, clock, *, max_wakes, window=60.0):
    """A fresh wake (a stand-in router process) sharing the on-disk breaker state + clock."""
    breaker = WakeBreaker(tmp_path, max_wakes=max_wakes, window=window, now=clock)
    agent, _ = build_wake(tmp_path, provider, breaker=breaker)
    return agent


def test_a_wake_burst_trips_self_declines_and_alerts_exactly_once(platform, tmp_path):
    """End to end: a runaway burst trips the breaker; the tripping (and every later) wake makes
    NO provider call, the loud alert posts exactly once, and the unseen message is left
    recoverable (its mark never advanced)."""
    clock = FakeClock()
    provider = CountingProvider()
    MarkStore(tmp_path).set(TIMELINE_UUID, M0)

    def wake_with(*page_messages):
        serve_messages(platform, page(*page_messages))
        return _wake_with_breaker(tmp_path, provider, clock, max_wakes=2).wake()

    # Two healthy wakes, each answering a new message → two provider calls.
    assert len(wake_with(message(uuid=M1, body="one"), message(uuid=M0, body="old"))) == 1
    assert len(wake_with(message(uuid=M2, body="two"), message(uuid=M1, body="one"))) == 1
    assert provider.prompts == [
        "[2026-06-04T00:00:00.000Z] john: one",
        "[2026-06-04T00:00:00.000Z] john: two",
    ]

    # The third wake would answer M3 — but it is over the cap in the window: TRIP, self-decline.
    assert wake_with(message(uuid=M3, body="three"), message(uuid=M2, body="two")) == []
    assert provider.prompts == [
        "[2026-06-04T00:00:00.000Z] john: one",
        "[2026-06-04T00:00:00.000Z] john: two",
    ]  # the model was NOT called
    assert MarkStore(tmp_path).get(TIMELINE_UUID) == M2  # M3 unseen → recoverable next healthy wake

    # A fourth, still-tripped wake also makes no provider call — and posts no second alert.
    assert wake_with(message(uuid=M3, body="three"), message(uuid=M2, body="two")) == []
    assert provider.prompts == [
        "[2026-06-04T00:00:00.000Z] john: one",
        "[2026-06-04T00:00:00.000Z] john: two",
    ]

    # Exactly one loud trip alert across the whole burst (the trip transition only).
    assert _alert_bodies(platform).count(_BREAKER_TRIP_ALERT) == 1


def test_the_breaker_auto_resets_and_resumes_after_the_burst_clears(platform, tmp_path):
    """Once the burst clears past the cooldown, the next wake auto-resets: it posts the recovery
    alert and resumes normal operation (answers the message it had been declining)."""
    clock = FakeClock()
    provider = CountingProvider()
    MarkStore(tmp_path).set(TIMELINE_UUID, M0)

    def wake_with(*page_messages):
        serve_messages(platform, page(*page_messages))
        return _wake_with_breaker(tmp_path, provider, clock, max_wakes=2).wake()

    # Drive a trip (two healthy, then the third trips).
    wake_with(message(uuid=M1, body="one"), message(uuid=M0, body="old"))
    wake_with(message(uuid=M2, body="two"), message(uuid=M1, body="one"))
    assert wake_with(message(uuid=M3, body="three"), message(uuid=M2, body="two")) == []
    assert provider.prompts == [
        "[2026-06-04T00:00:00.000Z] john: one",
        "[2026-06-04T00:00:00.000Z] john: two",
    ]

    # The burst stops; time passes past the window + cooldown. The next wake auto-resets and
    # answers M3, the message it had been declining.
    clock.advance(200)
    posted = wake_with(message(uuid=M3, body="three"), message(uuid=M2, body="two"))
    assert len(posted) == 1
    assert provider.prompts == [
        "[2026-06-04T00:00:00.000Z] john: one",
        "[2026-06-04T00:00:00.000Z] john: two",
        "[2026-06-04T00:00:00.000Z] john: three",
    ]  # resumed
    assert MarkStore(tmp_path).get(TIMELINE_UUID) == M3
    assert _alert_bodies(platform).count(_BREAKER_RESET_ALERT) == 1  # recovery alert posted once


def test_a_tripped_wake_alert_degrades_on_a_locked_timeline(platform, tmp_path):
    """The breaker's own alert post is best-effort: a locked timeline refusing it is swallowed,
    the wake still self-declines cleanly (no crash, no provider call)."""
    clock = FakeClock()
    provider = CountingProvider()
    # Pre-fill the window so the very next wake trips (cap 1; two recorded wakes already in window).
    pre = WakeBreaker(tmp_path, max_wakes=1, window=60.0, now=clock)
    pre.record_and_check(TIMELINE_UUID)
    pre.record_and_check(TIMELINE_UUID)  # now tripped on disk
    # The timeline refuses every post (locked).
    platform.post(f"/timelines/{TIMELINE_UUID}/messages").mock(
        return_value=httpx.Response(403, json=_locked_problem())
    )
    serve_messages(platform, page(message(uuid=M0, body="hi")))

    posted = _wake_with_breaker(tmp_path, provider, clock, max_wakes=1).wake()  # must not raise

    assert posted == []
    assert provider.prompts == []  # tripped → no provider call


def test_a_directly_constructed_wake_gets_a_default_breaker(platform, tmp_path):
    """A `WakeAgent` built without an explicit breaker still has one (the generous default),
    so the backstop is on by construction, not only via `from_env`."""
    agent, _ = build_wake(tmp_path)
    assert isinstance(agent.breaker, WakeBreaker)
    assert agent.breaker.max_wakes == 10 and agent.breaker.window == 60.0


# --- current-time grounding: the brief anchor + per-item timestamps ----------
#
# Every model call is grounded in time two ways: an absolute "now" at the head of the
# brief (`_now_line`), and a `[created_at]` stamp on every inbound item the agent
# perceives, which the model reads against that anchor to reason about an item's age.

# The created_at the wire fixtures above share, rendered as the agent perceives it.
TS = "2026-06-04T00:00:00.000Z"


def test_now_line_is_the_titlecased_utc_anchor():
    # `Current Time: 2026-06-21 17:09:49 UTC (+00:00, Sunday)` — Title Case label, absolute
    # UTC with an explicit offset, day-of-week, no trailing period (the anchor is a label, not
    # a sentence) — followed by a one-line UTC-conversion instruction (issue #180).
    line = _now_line()
    anchor, instruction = line.split("\n", 1)
    assert re.fullmatch(
        r"Current Time: \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC \(\+00:00, [A-Z][a-z]+day\)",
        anchor,
    )
    assert not anchor.endswith(".")
    # The instruction labels the clock UTC and tells the model to convert for a named locale,
    # so a bare UTC day/date is no longer parroted as if it were local (issue #180).
    assert "UTC" in instruction
    assert "convert" in instruction.lower()


def test_incoming_message_is_timestamped():
    msg = SimpleNamespace(
        created_at=TS,
        user=SimpleNamespace(handle="john"),
        content=SimpleNamespace(body="what is the current time?"),
    )
    assert _incoming_text(msg) == f"[{TS}] john: what is the current time?"


def test_incoming_asset_is_timestamped():
    asset = SimpleNamespace(
        created_at=TS,
        user=SimpleNamespace(handle="john"),
        content=SimpleNamespace(
            uuid="019e7780-7777-7aaa-8bbb-728394051627",
            description="",
            file=SimpleNamespace(filename="photo.png", byte_size=2048, content_type="image/png"),
        ),
    )
    assert _incoming_asset_text(asset).startswith(f"[{TS}] john posted a file")


def test_incoming_event_is_timestamped():
    event = SimpleNamespace(
        created_at=TS,
        webhook_endpoint=SimpleNamespace(uuid="019e7760-1111-7aaa-8bbb-1c2d3e4f5061"),
        content=SimpleNamespace(
            uuid="019e7761-2222-7bbb-8ccc-2d3e4f506172",
            content_type="application/json",
            payload='{"ok": true}',
        ),
    )
    text = _incoming_event_text(event)
    assert text.startswith(f"[{TS}] An inbound webhook was delivered")
    assert "was just delivered" not in text  # the now-redundant "just" is dropped


def test_activated_task_is_timestamped():
    # The task ITEM's created_at is its activation moment (≈ now), not when it was
    # scheduled — so the stamp reads consistent with every other inbound item.
    task = SimpleNamespace(
        created_at=TS,
        content=SimpleNamespace(
            uuid="019e7770-5555-7eee-8fff-506172839405",
            activate_at="2026-06-11T06:00:00+00:00",
            instructions="post the summary",
        ),
    )
    text = _activated_task_text(task)
    assert text.startswith(f"[{TS}] A task you scheduled has activated")
    assert "scheduled for 2026-06-11T06:00:00+00:00" in text  # complementary, retained


# --- read-speed pacing for AI↔AI conversations (issue #224) ------------------
#
# Before a wake answers a PEER AI's message it sleeps to simulate a human reading
# that message (`ReadPacer`), so an AI↔AI exchange is watchable and stays under the
# wake-breaker's trip line. Entirely receiver-side and derived from data the wake
# already fetches (the newest message's author `kind`, `body` length, `created_at`).
# Human messages are unaffected (instant). These tests inject a fake clock and a
# recording no-op sleep, so they assert the COMPUTED delay and never actually wait.

# A distinct peer AI (not the agent, not the human) — pacing's one target kind.
PEER_AI_UUID = "019e7756-9f60-7a80-93a4-6f7081920314"

# The fixtures' `created_at` is `2026-06-04T00:00:00.000Z`; a clock pinned to that same
# instant makes a message's age exactly 0, so the paced delay is the full read-time.
PACE_CREATED = datetime(2026, 6, 4, 0, 0, 0, tzinfo=timezone.utc)


class RecordingSleep:
    """A no-op stand-in for `time.sleep` that records every requested duration."""

    def __init__(self):
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


def _pace_msg(body, *, kind="ai", created_at="2026-06-04T00:00:00.000Z"):
    """A minimal message object with the three fields the pacer reads."""
    return SimpleNamespace(
        user=SimpleNamespace(kind=kind),
        content=SimpleNamespace(body=body),
        created_at=created_at,
    )


def _pacer(*, now=PACE_CREATED, **kwargs):
    """A `ReadPacer` with a fixed clock and a recording sleep; returns (pacer, sleep)."""
    sleep = RecordingSleep()
    return ReadPacer(clock=lambda: now, sleep=sleep, **kwargs), sleep


def peer_ai_message(*, uuid, body, created_at="2026-06-04T00:00:00.000Z"):
    """A wire message authored by a *different* AI peer — the kind pacing acts on."""
    return {
        "type": "message",
        "created_at": created_at,
        "user": {"uuid": PEER_AI_UUID, "handle": "briggs", "name": "Briggs", "kind": "ai"},
        "timeline": {"uuid": TIMELINE_UUID},
        "content": {"uuid": uuid, "body": body},
    }


# --- the ReadPacer math, in isolation (fake clock + recording sleep) ----------


def test_peer_ai_message_is_paced_for_its_read_time():
    """A peer AI's message → sleep(max(floor, chars/rate) - age); here age 0, 400 chars → 20s."""
    pacer, sleep = _pacer()
    slept = pacer.pace(_pace_msg("x" * 400))  # 400 / 20 chars-per-sec = 20s, above the 15s floor

    assert slept == 20.0
    assert sleep.calls == [20.0]


def test_the_floor_applies_to_a_very_short_message():
    """A one-word peer-AI reply reads in a blink, but the 15s floor keeps it human-paced."""
    pacer, sleep = _pacer()
    slept = pacer.pace(_pace_msg("ok"))  # 2 / 20 = 0.1s → clamped up to the 15s floor

    assert slept == 15.0
    assert sleep.calls == [15.0]


def test_delay_scales_with_message_length():
    """Twice the characters → twice the read-time (above the floor), a true length scaling."""
    short, short_sleep = _pacer()
    long, long_sleep = _pacer()

    assert short.pace(_pace_msg("x" * 400)) == 20.0  # 400 / 20
    assert long.pace(_pace_msg("x" * 800)) == 40.0  # 800 / 20 — proportionally longer
    assert short_sleep.calls == [20.0]
    assert long_sleep.calls == [40.0]


def test_age_is_subtracted_so_only_the_remainder_is_waited():
    """LOAD-BEARING `- age`: a message already 5s old owes only the remaining 15s of its 20s read."""
    pacer, sleep = _pacer(now=PACE_CREATED.replace(second=5))  # message aged 5s since it appeared
    slept = pacer.pace(_pace_msg("x" * 400))  # target 20s, age 5s → wait 15s

    assert slept == 15.0
    assert sleep.calls == [15.0]


def test_a_message_older_than_its_read_time_is_not_paced():
    """When age >= target the remainder clamps to 0 — no sleep (the 'quicker across timelines' case)."""
    pacer, sleep = _pacer(now=PACE_CREATED.replace(minute=5))  # 300s old, far past a 20s read
    slept = pacer.pace(_pace_msg("x" * 400))

    assert slept == 0.0
    assert sleep.calls == []  # never slept


def test_a_negative_age_is_clamped_so_the_delay_never_exceeds_the_read_time():
    """LOAD-BEARING clamp: a future-dated stamp / lagging box clock must not inflate the sleep."""
    # The box clock is 5 minutes behind the message's `created_at` → age is -300s. Unclamped,
    # `target - age` would be 320s; clamped, the message owes only its full 20s read-time.
    pacer, sleep = _pacer(now=PACE_CREATED - timedelta(minutes=5))
    slept = pacer.pace(_pace_msg("x" * 400))

    assert slept == 20.0  # target, never target + skew
    assert sleep.calls == [20.0]


def test_a_human_message_is_never_paced():
    """The `kind == 'ai'` gate is the whole opt-in: a human peer gets an instant reply."""
    pacer, sleep = _pacer()
    slept = pacer.pace(_pace_msg("What's the status?", kind="human"))

    assert slept == 0.0
    assert sleep.calls == []


def test_no_message_is_never_paced():
    """A wake with no message to answer (asset/task/webhook only) passes None → no sleep."""
    pacer, sleep = _pacer()

    assert pacer.pace(None) == 0.0
    assert sleep.calls == []


def test_a_disabled_pacer_never_sleeps():
    """`enabled=False` (the kill switch) short-circuits even a long peer-AI message."""
    pacer, sleep = _pacer(enabled=False)

    assert pacer.pace(_pace_msg("x" * 400)) == 0.0
    assert sleep.calls == []


def test_a_nonpositive_rate_degrades_to_the_floor():
    """A misconfigured rate of 0 can't divide; it degrades to 'always the floor', never a crash."""
    pacer, sleep = _pacer(chars_per_sec=0)

    assert pacer.pace(_pace_msg("x" * 400)) == 15.0  # falls back to the floor, no ZeroDivisionError
    assert sleep.calls == [15.0]


# --- pacing wired through a real wake (the reply path) ------------------------


def test_a_wake_paces_before_answering_a_peer_ai(platform, tmp_path):
    """End-to-end: a peer AI's message is read-paced, then answered exactly as today."""
    serve_messages(platform, page(peer_ai_message(uuid=M0, body="x" * 400)))
    pacer, sleep = _pacer()
    agent, provider = build_wake(tmp_path, pacer=pacer)

    posted = agent.wake()

    assert sleep.calls == [20.0]  # paced the peer AI's 400-char message
    assert len(posted) == 1  # then replied as normal
    assert len(provider.prompts) == 1


def test_a_wake_answering_a_human_does_not_pace(platform, tmp_path):
    """The existing human path is byte-for-byte unchanged — no sleep, instant reply."""
    serve_messages(platform, page(message(uuid=M0, body="What's the status?")))  # john, human
    pacer, sleep = _pacer()
    agent, provider = build_wake(tmp_path, pacer=pacer)

    agent.wake()

    assert sleep.calls == []


def test_own_newest_message_is_not_paced(platform, tmp_path):
    """If the newest answered message is the agent's own, it self-filters out → no pacing."""
    MarkStore(tmp_path).set(TIMELINE_UUID, M0)
    # Newer than the mark, but authored by the agent (mine=True) — self-filtered before the gate.
    serve_messages(
        platform, page(message(uuid=M1, body="x" * 400, mine=True), message(uuid=M0, body="old"))
    )
    pacer, sleep = _pacer()
    agent, provider = build_wake(tmp_path, pacer=pacer)

    agent.wake()

    assert sleep.calls == []  # nothing non-self to react to → no read-pace
    assert provider.prompts == []  # and the own message is self-skipped, not answered


def test_a_task_only_wake_does_not_pace(platform, tmp_path):
    """Pacing is message-scoped: an asset/task/webhook-only wake never sleeps."""
    MarkStore(tmp_path).set(TIMELINE_UUID, M0)  # messages already caught up → nothing to answer
    serve_messages(platform, page(message(uuid=M0, body="old")))
    serve_tasks(platform, task_page(task(uuid=T0, instructions="post the daily summary")))
    pacer, sleep = _pacer()
    agent, provider = build_wake(tmp_path, pacer=pacer)

    agent.wake()

    assert sleep.calls == []  # the task path never routes through the paced message reply


def test_a_signed_probe_is_not_paced(platform, tmp_path):
    """A NOC probe must stay a sub-second token-free ack — pacing is skipped even from an AI peer."""
    body = f"NOC message-seam probe — please disregard.\n{probe_marker()}"
    serve_messages(platform, page(peer_ai_message(uuid=M0, body=body)))  # AI-authored probe
    pacer, sleep = _pacer()
    agent, provider = build_wake(tmp_path, pacer=pacer, probe_secret=PROBE_SECRET)

    posted = agent.wake()

    assert sleep.calls == []  # the heartbeat is never delayed by the read-pace
    assert len(posted) == 1  # the ack still went out
    assert provider.prompts == []  # still token-free — the model never ran
    sent = platform.post(f"/timelines/{TIMELINE_UUID}/messages").calls.last.request
    assert json.loads(sent.content) == {"message": {"body": f"BCNOC1-ACK {PROBE_NONCE}"}}


def test_a_probe_earlier_in_the_batch_still_skips_pacing(platform, tmp_path):
    """A probe that is NOT the newest item still skips pacing — its ack must stay sub-second.

    The sleep precedes `_act_on`, which acks every message in the batch; a probe older than a
    real peer-AI message would otherwise have its ack delayed by pacing the newer message. So
    *any* probe in the batch short-circuits pacing for the wake.
    """
    # A mark older than both served messages, so the batch is [probe (older), peer-AI (newer)].
    MarkStore(tmp_path).set(TIMELINE_UUID, "019e7740-0000-7000-8000-000000000000")
    body = f"NOC message-seam probe — please disregard.\n{probe_marker()}"
    serve_messages(
        platform,
        page(
            peer_ai_message(uuid=M1, body="a real reply that would otherwise be paced"),
            peer_ai_message(uuid=M0, body=body),  # older — the probe, not the newest item
        ),
    )
    pacer, sleep = _pacer()
    agent, provider = build_wake(tmp_path, pacer=pacer, probe_secret=PROBE_SECRET)

    agent.wake()

    assert sleep.calls == []  # a probe anywhere in the batch → no pacing, heartbeat preserved


def test_pacing_never_crashes_the_wake_on_a_bad_timestamp(platform, tmp_path):
    """A malformed `created_at` degrades pacing to no delay — it must never kill the wake (B2)."""
    serve_messages(
        platform, page(peer_ai_message(uuid=M0, body="x" * 400, created_at="not-a-timestamp"))
    )
    pacer, sleep = _pacer()
    agent, provider = build_wake(tmp_path, pacer=pacer)

    posted = agent.wake()  # must not raise despite the unparseable stamp

    assert sleep.calls == []  # pacing degraded to no delay
    assert len(posted) == 1  # the wake still answered the message normally


# --- env tunables and the parse helper ---------------------------------------


def test_pace_is_enabled_by_default_and_env_can_disable_it(monkeypatch):
    """`HARNESS_PACE_ENABLED` is on unless explicitly off (mirrors HARNESS_ONBOARD)."""
    monkeypatch.delenv("HARNESS_PACE_ENABLED", raising=False)
    assert _pace_enabled_from_env() is True  # unset → on

    monkeypatch.setenv("HARNESS_PACE_ENABLED", "false")
    assert _pace_enabled_from_env() is False

    monkeypatch.setenv("HARNESS_PACE_ENABLED", "off")
    assert _pace_enabled_from_env() is False

    monkeypatch.setenv("HARNESS_PACE_ENABLED", "")  # blank → still on (only explicit off disables)
    assert _pace_enabled_from_env() is True


def test_pace_env_tunables_override_the_defaults(monkeypatch):
    """`HARNESS_PACE_CHARS_PER_SEC`/`_FLOOR_SECONDS` override the real defaults; blank → default."""
    monkeypatch.delenv("HARNESS_PACE_CHARS_PER_SEC", raising=False)
    monkeypatch.delenv("HARNESS_PACE_FLOOR_SECONDS", raising=False)
    assert _pace_chars_per_sec_from_env() == 20.0  # the real default
    assert _pace_floor_seconds_from_env() == 15.0

    monkeypatch.setenv("HARNESS_PACE_CHARS_PER_SEC", "50")
    monkeypatch.setenv("HARNESS_PACE_FLOOR_SECONDS", "3")
    assert _pace_chars_per_sec_from_env() == 50.0
    assert _pace_floor_seconds_from_env() == 3.0

    # from_env threads them into a live pacer.
    pacer = ReadPacer.from_env(clock=lambda: PACE_CREATED, sleep=RecordingSleep())
    assert pacer.chars_per_sec == 50.0
    assert pacer.floor_seconds == 3.0


def test_a_disabled_env_makes_from_env_pace_nothing(monkeypatch):
    """`HARNESS_PACE_ENABLED=false` → `from_env` builds a pacer that never sleeps."""
    monkeypatch.setenv("HARNESS_PACE_ENABLED", "false")
    sleep = RecordingSleep()
    pacer = ReadPacer.from_env(clock=lambda: PACE_CREATED, sleep=sleep)

    assert pacer.pace(_pace_msg("x" * 400)) == 0.0
    assert sleep.calls == []


def test_parse_created_at_handles_z_suffix_and_naive_stamps():
    """The ISO parse normalizes a `Z` suffix (3.10-safe) and assumes UTC for a naive stamp."""
    z = _parse_created_at("2026-06-04T00:00:00.000Z")
    assert z == PACE_CREATED and z.tzinfo is not None

    offset = _parse_created_at("2026-06-04T00:00:00+00:00")
    assert offset == PACE_CREATED

    naive = _parse_created_at("2026-06-04T00:00:00")  # no offset → assumed UTC, aware
    assert naive == PACE_CREATED and naive.tzinfo is timezone.utc
