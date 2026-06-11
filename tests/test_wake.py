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

from basecradle_harness import Harness, MarkStore, Message, SeenStore, WakeAgent
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
        # By default a timeline has no posted assets, no inbound webhook deliveries,
        # and no activated tasks, so a wake's reconciliation of them is a clean no-op.
        # Tests that exercise those paths override these with the matching `serve_*`.
        router.get("/assets").mock(return_value=httpx.Response(200, json=asset_page()))
        router.get("/webhook_events").mock(return_value=httpx.Response(200, json=event_page()))
        router.get("/tasks").mock(return_value=httpx.Response(200, json=task_page()))
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
    assert any("john: hi" in p for p in provider.prompts)
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
    monkeypatch.setenv("AI_PROVIDER_MODEL", "gpt-4o")
    monkeypatch.setenv("AI_PROVIDER_API_KEY", "sk-test-key")
    monkeypatch.setenv("HARNESS_HOME", str(tmp_path))
    monkeypatch.setenv("HARNESS_ONBOARD", "0")
    monkeypatch.delenv("BASECRADLE_TIMELINE", raising=False)
    monkeypatch.delenv("BASECRADLE_MESSAGE", raising=False)
    monkeypatch.delenv("BASECRADLE_EVENT", raising=False)
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


def test_main_returns_nonzero_when_home_is_missing(platform, wake_env, monkeypatch):
    """A hard config failure (no HARNESS_HOME) exits non-zero so the router reports it."""
    monkeypatch.delenv("HARNESS_HOME", raising=False)

    assert main(["--timeline", TIMELINE_UUID]) == 1


def test_main_returns_nonzero_on_missing_provider_config(platform, wake_env, monkeypatch):
    monkeypatch.delenv("AI_PROVIDER_MODEL", raising=False)

    assert main(["--timeline", TIMELINE_UUID]) == 1
