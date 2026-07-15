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
import contextlib
import hmac
import io
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from importlib import metadata
from types import SimpleNamespace

import httpx
import pytest
import respx
from basecradle import BaseCradle

from basecradle_harness import (
    BreakerDecision,
    Claim,
    ClaimStore,
    Harness,
    MarkStore,
    Message,
    MessagesTool,
    ReadPacer,
    SeenStore,
    Tool,
    WakeAgent,
    WakeBreaker,
    _wake,
    install,
)
from basecradle_harness._basecradle import _incoming_text, _parse_created_at
from basecradle_harness._messages import ToolCall
from basecradle_harness._wake import (
    _activated_task_text,
    _incoming_asset_text,
    _incoming_event_text,
    _now_line,
    _pace_chars_per_sec_from_env,
    _pace_enabled_from_env,
    _pace_floor_seconds_from_env,
    _pace_max_builds_from_env,
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


def _is_counter(m) -> bool:
    """A live step-counter note the engine injects before each provider call (issue #243).

    Matched by its `Step N of M` line — never the brief's own `Current Time:` anchor or its
    `Step budget:` statement, whose wording is deliberately close but carries no `N of M` (so a
    `\\n\\nStep ` substring check would wrongly swallow the whole onboarding brief).
    """
    return m.role == "system" and bool(m.content) and bool(re.search(r"Step \d+ of \d+", m.content))


def _convo(messages: list) -> list:
    """The turns the model saw with the injected step-counter notes filtered out."""
    return [m for m in messages if not _is_counter(m)]


def test_is_counter_does_not_swallow_the_onboarding_brief():
    """Guard: the brief starts with `Current Time:` and now carries a `Step budget:` line, so a
    naive `\\n\\nStep ` check would misclassify the whole brief as a counter note and filter it
    out — weakening every wake assertion. The `Step N of M` discriminator must not match it."""
    from basecradle_harness import compose_brief, render_budget

    brief = compose_brief(
        now=_now_line(),
        budget=render_budget(24),
        initialize="How to operate here.",
        manifest="Your active tools right now:\n- memory",
        dashboard="DASH",
        system_prompt="You are Nova.",
    )
    assert "Step budget:" in brief  # the collision-prone content is actually present
    assert not _is_counter(Message(role="system", content=brief))  # …yet the brief is not filtered
    # A real counter note, by contrast, IS matched.
    assert _is_counter(Message(role="system", content="Current Time: X\n\nStep 3 of 24."))


class _NoopTool(Tool):
    """A trivial tool, registered so the engine offers a non-empty tool set (`specs` non-None).

    A provider that keys its reserve behavior on ``tools is None`` needs the budgeted loop to
    pass a real tool set, which only happens when the registry is non-empty (issue #243 tests).

    It **counts its runs**, because the recovery's central promise is negative — a resumed turn must
    not fire the dead turn's tools a second time — and a promise about something *not* happening can
    only be tested by something that would have noticed (issue #297).
    """

    name = "noop"
    description = "Does nothing."

    def __init__(self) -> None:
        self.runs = 0

    def run(self, **kwargs) -> str:
        self.runs += 1
        return "ok"


class CountingProvider:
    """A canned brain that records every call, so we can assert when it is (not) used.

    **It speaks the way every agent speaks now** (the Unspoken Channel, issue #293): it calls the
    `messages` tool with its canned text, and then ends the turn with a narration nobody posts.
    That is deliberate and it is the point — the harness posts nothing of its own any more, so a
    canned brain that only *returns text* is a canned brain that says nothing at all. Wiring the
    default this way means the whole reconcile suite (messages, assets, tasks, webhook deliveries)
    exercises the real production path: engage → tool call → post → narrate.

    `speak=False` gives the other shape: a model that thinks and stays silent. Two kinds of test
    need it — the ones pinning that silence posts nothing, and the pacing/rebuild ones, where a
    build that ran a tool is (correctly) never rolled back.
    """

    # The labels every shipped adapter carries (`describe_provider`), so a wake's bookend lines
    # name a provider and a model here exactly as they do in production.
    provider = "openai"
    model = "gpt-4o"

    def __init__(self, text="Hello, John.", *, speak=True, narration="Answered him."):
        self.text = text
        self.speak = speak
        self.narration = narration
        self.prompts: list[str] = []
        self.last_messages: list = []
        # A snapshot of the images on the last turn at *chat time* — captured before the
        # session evicts the pixels, so a test can assert an asset image was actually
        # presented to the model (the live object's `.images` is emptied after the turn).
        self.last_images: list = []
        self._calls = 0

    def chat(self, messages, tools=None):
        self._calls += 1
        self.last_messages = list(messages)
        # The engine appends a step-counter note as the last turn, so an asset's image no longer
        # rides messages[-1]; scan all turns for the pixels.
        self.last_images = [img for m in messages for img in m.images]
        last = _convo(messages)[-1]
        if last.role != "user":
            # The tool result is back — settle the turn with the (unspoken) narration. Only the
            # *first* call of a turn is a prompt the wake engaged us on, which is why `prompts`
            # is appended above and not here: it stays "the items the model was shown".
            return Message.assistant(content=self.narration)
        self.prompts.append(last.content)
        if not self.speak:
            return Message.assistant(content=self.text)  # a silent turn: thought, said nothing
        return Message.assistant(
            tool_calls=[
                ToolCall(
                    id=f"call-{self._calls}",
                    name="messages",
                    arguments={"action": "create", "body": self.text},
                )
            ]
        )


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
    """Drive the (newest-first) message list endpoint; the LAST page repeats for every read.

    A #226 message wake reads the list several times per turn — the initial gather, then the
    Loop-1 settle re-check and the Loop-2 staleness re-check(s) — so a single fixed list must
    satisfy an unbounded number of reads. Each given page is served once in order, then the
    **last** page repeats forever; a re-read after the mark advanced simply yields nothing
    newer (`_messages_since` filters past the mark). Pass one page for a steady list; the
    multi-page form scripts an early read differently from the settled tail. For a mid-wake
    *change* (a message landing during pacing/generation) use `ScriptedMessages`, which mutates
    a single served list so the exact read count never has to be counted.
    """
    queue = [httpx.Response(200, json=p) for p in pages]

    def _serve(request):
        return queue.pop(0) if len(queue) > 1 else queue[0]

    platform.get("/messages").mock(side_effect=_serve)


def _posts(platform):
    """Every message body this agent POSTed, in order — read from the router's global call log.

    Deliberately **not** `platform.post(...).calls`: calling `router.post(...)` mid-test registers
    a *fresh, unmocked* route that then swallows the next real POST (it cost an hour). Reading the
    call log is inert — it registers nothing and can be asserted before and after a wake alike.
    """
    return [
        json.loads(call.request.content)["message"]["body"]
        for call in platform.calls
        if call.request.method == "POST" and call.request.url.path.endswith("/messages")
    ]


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


def _is_brief(m) -> bool:
    """A persistent-brief system turn, recognized by `initialize.md`'s heading."""
    return m.role == "system" and "How to operate here" in (m.content or "")


def _brief_shown(provider):
    """The brief turns the model was shown on its last call.

    The only place the brief is observable: it is **ephemeral** (issue #275) — spliced into the
    message list handed to the provider and never written to the transcript — so a test that
    looks for it in `history` is asserting the bug, not the contract.
    """
    return [m for m in provider.last_messages if _is_brief(m)]


def _brief_turns(agent):
    """Brief-shaped system turns *persisted* in the transcript — always empty (issue #275).

    The per-wake brief is a snapshot of a moment (current time, step budget, live dashboard), so
    a persisted copy is a stale duplicate the model re-reads — and re-pays for — on every later
    turn. Kept as a named helper because "this stays zero" is the invariant worth asserting.
    """
    history = agent.harness.session(agent.source).history
    return [m for m in history if _is_brief(m)]


def build_wake(home, provider=None, *, system_prompt=None, onboard=False, tools=None, **kwargs):
    """A fresh WakeAgent over `home` — a stand-in for one router-spawned process.

    The agent gets a `MessagesTool` by default, because since issue #293 that is the **only** way
    an agent can speak: a harness with no messages tool is a mute agent, and every test that
    asserts a post would be asserting the absence of a tool rather than the presence of a
    decision. `tools=[]` builds the mute agent deliberately.
    """
    provider = provider or CountingProvider()
    client = BaseCradle(token=FAKE_TOKEN)
    harness = Harness(
        provider,
        system_prompt=system_prompt,
        home=home,
        tools=[MessagesTool()] if tools is None else tools,
    )
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


def test_multiple_unseen_messages_get_one_batched_reply_oldest_first(platform, tmp_path):
    """#226 many-to-one: N unseen peer messages → ONE reply, seeded oldest-first as one turn.

    The pre-#226 path looped a reply per message (N → N); the batch reply gathers them all,
    renders them oldest-first into a single turn, and answers once. The mark still advances past
    the newest, and every message in the batch is claimed."""
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

    assert len(posted) == 1  # ONE batched reply to both, not one-per-message
    assert provider.prompts == [
        "[2026-06-04T00:00:00.000Z] john: first\n[2026-06-04T00:00:00.000Z] john: second"
    ]  # both messages, oldest-first, in one turn
    assert MarkStore(tmp_path).get(TIMELINE_UUID) == M2  # mark past the newest
    assert ClaimStore(tmp_path).claim(TIMELINE_UUID, M1, kind="messages") is False  # both claimed
    assert ClaimStore(tmp_path).claim(TIMELINE_UUID, M2, kind="messages") is False


# --- idempotency across a crash / retry mid-batch ----------------------------


def test_a_batch_generation_crash_is_re_driven_and_the_peers_are_answered(platform, tmp_path):
    """Issue #285: a wake that dies **inside the model call** no longer drops the peers' messages.

    This test used to pin the opposite, and its old name said so — *"does not reprocess the
    batch"*. The batch was claimed **and marked seen** before the model call, so a crash during
    generation advanced the mark over messages that were never answered, and **no future wake
    would ever look at them again**. That was a deliberate at-most-once trade (a dropped reply
    beats a backlog re-answered with side effects), and #285 retires it: the trade is unnecessary,
    because the transcript says which of the two actually happened.

    The provider raises a non-`EngineError` — a process dying mid-generation. Nothing ran and
    nothing posted, so the crash is *provably side-effect-free* and the messages are re-driven.
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

    class DiesGenerating:
        def chat(self, messages, tools=None):
            raise RuntimeError("boom mid-generation")

    agent, _ = build_wake(tmp_path, DiesGenerating())
    with pytest.raises(RuntimeError):
        agent.wake()

    # The mark did NOT advance over the unanswered messages — this is the whole fix. It still
    # sits where it started, so the next wake reads them again instead of never seeing them.
    assert MarkStore(tmp_path).get(TIMELINE_UUID) == M0
    # And they are left `in-flight` — claimed, but not settled: the record that says a wake took
    # them and never finished, which is what makes them recoverable rather than lost.
    claims = ClaimStore(tmp_path)
    assert claims.read(TIMELINE_UUID, M1, kind="messages").phase == "in-flight"
    assert claims.read(TIMELINE_UUID, M2, kind="messages").phase == "in-flight"

    # A retry (fresh process, same home) re-drives them: the dead wake's pid is gone, its
    # transcript shows no tool ran and no reply was produced, so answering is safe.
    serve_messages(platform, page(message(uuid=M2, body="second"), message(uuid=M1, body="first")))
    retry, retry_provider = build_wake(tmp_path)
    posted = retry.wake()

    assert retry_provider.prompts, "the re-driving wake must actually engage the model"
    assert len(posted) == 1  # one batched reply answering both, exactly as a healthy wake would
    assert MarkStore(tmp_path).get(TIMELINE_UUID) == M2  # now settled, the mark may pass
    assert claims.read(TIMELINE_UUID, M1, kind="messages").phase == "done"
    assert claims.read(TIMELINE_UUID, M2, kind="messages").phase == "done"


# --- B2: a wake never crashes on an SDK / engine error -----------------------


def _locked_problem():
    """An RFC 9457 problem doc for a locked timeline — what the post is refused with."""
    return {
        "status": 403,
        "code": "timeline_locked",
        "title": "Timeline Locked",
        "detail": "This timeline is locked and is not accepting new content.",
    }


def test_a_locked_timeline_refusal_reaches_the_model_instead_of_crashing(platform, tmp_path):
    """B2: a post refused by a locked timeline degrades — and now the *model* is the one told.

    The refusal used to land on the harness (which posted the reply) and was recorded as a note in
    the transcript. Since the agent speaks for itself (issue #293), the refusal comes back as the
    result of its own `messages` tool call: it learns, mid-turn, that it cannot speak here and can
    act on that. Still no crash, still marked seen (no reprocess loop), still exit 0.
    """
    MarkStore(tmp_path).set(TIMELINE_UUID, M0)
    serve_messages(platform, page(message(uuid=M1, body="hi"), message(uuid=M0, body="old")))
    platform.post(f"/timelines/{TIMELINE_UUID}/messages").mock(
        return_value=httpx.Response(403, json=_locked_problem())
    )
    agent, provider = build_wake(tmp_path)

    posted = agent.wake()  # must not raise

    assert posted == []  # the agent tried to speak; the platform refused
    assert provider.prompts == ["[2026-06-04T00:00:00.000Z] john: hi"]  # the model still ran
    assert MarkStore(tmp_path).get(TIMELINE_UUID) == M1  # marked seen despite the failed post
    # The refusal reached the model as its tool result, in words it can act on.
    results = [t.content for t in agent.harness.session(agent.source).history if t.role == "tool"]
    assert any("Couldn't create a message" in r for r in results)


def test_main_returns_zero_on_a_locked_timeline(platform, wake_env):
    """B2 end to end: the entrypoint exits 0 on a locked timeline — no traceback escapes.

    The model replies, but the timeline refuses the post (`timeline_locked`); the degrade
    keeps the wake clean and the exit code 0, so the router sees success, not a crash."""
    _serve_openai_and_messages(platform, page(message(uuid=M0, body="status?")))
    platform.post(f"/timelines/{TIMELINE_UUID}/messages").mock(
        return_value=httpx.Response(403, json=_locked_problem())
    )

    assert main(["--timeline", TIMELINE_UUID]) == 0


def test_the_reserve_summary_is_unspoken_not_posted(platform, tmp_path, caplog):
    """The step-cap report goes to the **operator**, never to the peers (issues #243, #293).

    The reserve summary is the model's own honest progress report — "what I completed, what
    remains, what the next turn should do". That is a note addressed to the operator and to its own
    next turn; it was never something a peer asked to read, and it used to be posted to them
    anyway. It is now unspoken: journaled (`kind=reserve`), and nothing reaches the timeline.
    """
    serve_messages(platform, page(message(uuid=M0, body="do something complicated")))

    class LoopingThenSummary:
        """Calls a tool while budgeted; writes a progress report once tools are withheld."""

        def chat(self, messages, tools=None):
            if tools is None:  # the out-of-budget reserve call
                return Message.assistant(content="I researched the topic but ran out of steps.")
            return Message.assistant(tool_calls=[ToolCall(id="call_1", name="noop", arguments={})])

    client = BaseCradle(token=FAKE_TOKEN)
    harness = Harness(LoopingThenSummary(), tools=[_NoopTool()], home=tmp_path, max_steps=2)
    agent = WakeAgent(harness, timeline=TIMELINE_UUID, client=client, onboard=False)

    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        posted = agent.wake()  # must not raise

    assert posted == []
    assert _posts(platform) == []  # the peers were told nothing — it was not for them
    unspoken = _line(caplog, "unspoken")
    assert "I researched the topic but ran out of steps." in unspoken  # the operator has it, whole
    assert agent.marks.get(TIMELINE_UUID) == M0  # still marked seen — no reprocess


def test_reserve_call_failure_degrades_to_the_canned_note(platform, tmp_path, caplog):
    """The canned "I got stuck" note survives only as the fallback-of-the-fallback: the
    reserve summary call itself erroring (issue #243). It is unspoken too (#293) — a fact about
    the machinery giving up, which is the operator's business and nobody else's."""
    serve_messages(platform, page(message(uuid=M0, body="do something complicated")))

    class LoopingThenBoom:
        """Never settles, and the out-of-budget reserve call blows up too."""

        def chat(self, messages, tools=None):
            if tools is None:
                raise RuntimeError("reserve model call failed")
            return Message.assistant(tool_calls=[ToolCall(id="call_1", name="noop", arguments={})])

    client = BaseCradle(token=FAKE_TOKEN)
    harness = Harness(LoopingThenBoom(), tools=[_NoopTool()], home=tmp_path, max_steps=2)
    agent = WakeAgent(harness, timeline=TIMELINE_UUID, client=client, onboard=False)

    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        posted = agent.wake()  # must not raise

    assert posted == []
    assert _posts(platform) == []
    assert "got stuck" in _line(caplog, "unspoken")
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
    first, _ = build_wake(tmp_path, CountingProvider(speak=False))  # silent: a clean text turn
    first.wake()

    serve_messages(
        platform,
        page(message(uuid=M1, body="what did I say?"), message(uuid=M0, body="remember Ruby")),
    )
    second, provider = build_wake(tmp_path, CountingProvider(speak=False))
    second.wake()

    # The model saw the earlier exchange (loaded from disk) in front of the new turn.
    roles_and_text = [(m.role, m.content) for m in _convo(provider.last_messages)]
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

    assert len(posted) == 1  # one batched reply to everything since our last post
    assert provider.prompts == [
        "[2026-06-04T00:00:00.000Z] john: a follow-up\n[2026-06-04T00:00:00.000Z] john: and another"
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
        "[2026-06-04T00:00:00.000Z] john: the trigger\n[2026-06-04T00:00:00.000Z] john: newest"
    ]  # M1 forward, batched into one turn
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
    agent, provider = build_wake(tmp_path, CountingProvider(speak=False))

    agent.wake()

    context = [(m.role, m.content) for m in _convo(provider.last_messages)]
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


def test_activated_task_is_claimed_before_acting_so_a_re_entrant_wake_cannot_refire(
    platform, tmp_path
):
    """THE BUG-1 REGRESSION, on the guard that actually carries it: **the claim, not the record.**

    The monkey pile-up was an *act-then-record* gap: the task stayed `activated`, an action that
    re-woke the agent (a generated image posts an `asset.created`) re-surfaced it, and the
    unrecorded task fired again, and again. The fix at the time recorded it seen *before* acting —
    and that is what issue #289 removes, because the seen-set entry then landed before the model was
    ever called, so a wake killed in between filed a task nobody ever ran as done, forever.

    What stops the re-fire is the claim, and it always was: it lands before the turn, it is an
    atomic exclusive create, and a re-entrant wake that finds it simply skips the task. This pins
    exactly that — a second wake arriving *while the first still holds the task in flight* never
    touches it — with no seen-set entry anywhere in sight.
    """
    serve_messages(platform, page(), page())
    serve_tasks(
        platform,
        task_page(task(uuid=T0, instructions="generate a monkey and post it")),
        task_page(task(uuid=T0, instructions="generate a monkey and post it")),
    )
    live_wake_owning(tmp_path, T0, kind="tasks")  # a concurrent wake is mid-turn on this task

    intruder, brain = build_wake(tmp_path)
    assert intruder.wake() == []
    assert brain.prompts == []  # the model was never consulted: the task is already owned
    assert T0 not in SeenStore(tmp_path).all(TIMELINE_UUID, kind="tasks")  # and never recorded


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


class _TextOnlyProvider(CountingProvider):
    """A `CountingProvider` that answers the vision capability with a definite *no* (issue #228)."""

    def supports_vision(self):
        return False


def test_a_posted_image_degrades_to_text_for_a_non_vision_model(platform, tmp_path, caplog):
    """A model with no vision input is never blind-sent pixels: the image is swapped for its text
    description, so a text-only model reads *what* was shared rather than a "Looking at it now"
    caption for a picture it silently drops. This is the reachable @glm-5.2 case (issue #228)."""
    MarkStore(tmp_path).set(TIMELINE_UUID, A0, kind="assets")
    serve_messages(platform, page())
    serve_assets(platform, asset_page(asset(uuid=A1, filename="diagram.png")))
    agent, provider = build_wake(tmp_path, provider=_TextOnlyProvider())

    with caplog.at_level(logging.WARNING, logger="basecradle_harness"):
        posted = agent.wake()

    assert len(posted) == 1  # acted on, gracefully
    assert provider.last_images == []  # NOT shown the pixels the model can't take
    assert "diagram.png" in provider.prompts[0]  # the file is described instead
    assert MarkStore(tmp_path).get(TIMELINE_UUID, kind="assets") == A1


def test_the_image_degrade_is_logged_loudly(platform, tmp_path, caplog):
    """A silent degrade is a defect (issue #293): the swap leaves a WARNING naming the asset and the
    model, greppable in the box's Live Tail (the agent identifier rides journald, not the message)."""
    MarkStore(tmp_path).set(TIMELINE_UUID, A0, kind="assets")
    serve_messages(platform, page())
    serve_assets(platform, asset_page(asset(uuid=A1, filename="diagram.png")))
    agent, _ = build_wake(tmp_path, provider=_TextOnlyProvider())

    with caplog.at_level(logging.WARNING, logger="basecradle_harness"):
        agent.wake()

    line = _line(caplog, "image degraded to text")
    assert "diagram.png" in line
    assert "model has no vision input" in line
    assert TIMELINE_UUID in line


def test_a_vision_capable_model_is_still_shown_the_image(platform, tmp_path, caplog):
    """The gate fails toward perception: a model that answers vision=True keeps seeing the picture,
    and no degrade line is logged — the change is confined to the definite no-vision case."""
    MarkStore(tmp_path).set(TIMELINE_UUID, A0, kind="assets")
    serve_messages(platform, page())
    serve_assets(platform, asset_page(asset(uuid=A1, filename="diagram.png")))

    class _Sighted(CountingProvider):
        def supports_vision(self):
            return True

    agent, provider = build_wake(tmp_path, provider=_Sighted())

    with caplog.at_level(logging.WARNING, logger="basecradle_harness"):
        agent.wake()

    assert len(provider.last_images) == 1  # shown, exactly as before the gate existed
    assert not [m for m in _lines(caplog) if m.startswith("image degraded to text")]


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


def _responses_payload(output):
    """One `/responses` body carrying `output` — the Responses surface's reply envelope."""
    return {
        "id": "resp-wake",
        "object": "response",
        "created_at": 0,
        "model": "gpt-4o",
        "output": output,
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
    }


def _serve_openai_and_messages(platform, *pages, speak=True, body="On it."):
    """The model's wire replies plus the message pages — enough for `main` to run live.

    Wake mode runs the default @jt stack — the ``openai`` SDK on the **Responses** surface — so the
    model call lands on ``/responses`` (SDK-validated body), not ``/chat/completions``.

    **It speaks the way a real agent speaks** (issue #293): the first call returns a `function_call`
    item for the `messages` tool, the second settles on plain text. So these end-to-end tests drive
    the whole production stack — env → resolved tools → the Responses wire → tool dispatch → the
    post — rather than a plain-text turn that, since the inversion, would reach nobody. `speak=False`
    returns text on the first call: an agent that read, thought, and chose silence.
    """
    settle = [
        {
            "id": "msg-wake",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Answered him.", "annotations": []}],
        }
    ]
    if not speak:
        platform.post("https://api.openai.com/v1/responses").mock(
            return_value=httpx.Response(200, json=_responses_payload(settle))
        )
        serve_messages(platform, *pages)
        return
    call = [
        {
            "id": "fc-wake",
            "type": "function_call",
            "status": "completed",
            "call_id": "call-wake-1",
            "name": "messages",
            "arguments": json.dumps({"action": "create", "body": body}),
        }
    ]
    # The queue is per-call: the tool call first, then the settling narration — and the settle
    # repeats, so a wake with several turns (a task plus a message, say) never runs out of replies.
    replies = [
        httpx.Response(200, json=_responses_payload(call)),
        httpx.Response(200, json=_responses_payload(settle)),
    ]

    def _serve(request):
        return replies.pop(0) if len(replies) > 1 else replies[0]

    platform.post("https://api.openai.com/v1/responses").mock(side_effect=_serve)
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


@contextlib.contextmanager
def _isolated_root_logging():
    """Run with the root logger reset (no handlers) and restore it afterward.

    `_configure_logging` only installs a handler when the root logger has none — but under
    pytest the logging plugin keeps a capture handler on root, which would make the CLI's
    `logging.basicConfig` a silent no-op. Clearing root here reproduces the *production*
    starting state (a fresh process on Python's last-resort handler), so the test proves the
    real behavior, then restores pytest's handlers so log capture keeps working.
    """
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    root.handlers = []
    try:
        yield root
    finally:
        root.handlers, root.level = saved_handlers, saved_level


def test_configure_logging_installs_an_info_handler_when_unconfigured(monkeypatch):
    """The fix (issue #248): an unconfigured process gets an INFO stderr handler by default."""
    from basecradle_harness._basecradle import _configure_logging

    monkeypatch.delenv("HARNESS_LOG_LEVEL", raising=False)
    with _isolated_root_logging() as root:
        _configure_logging()
        assert root.handlers  # a handler was installed where there was none
        assert root.level == logging.INFO  # …at INFO, so the ledger's INFO lines pass


def test_configure_logging_honors_the_log_level_env_override(monkeypatch):
    """`HARNESS_LOG_LEVEL` tunes verbosity (name or number); the INFO default is only a default."""
    from basecradle_harness._basecradle import _configure_logging

    monkeypatch.setenv("HARNESS_LOG_LEVEL", "warning")
    with _isolated_root_logging() as root:
        _configure_logging()
        assert root.level == logging.WARNING


def test_configure_logging_leaves_an_already_configured_root_untouched():
    """An embedding application's own logging setup wins — the CLI never hijacks it."""
    from basecradle_harness._basecradle import _configure_logging

    with _isolated_root_logging() as root:
        sentinel = logging.NullHandler()
        root.handlers = [sentinel]
        _configure_logging()
        assert root.handlers == [sentinel]  # not replaced, not appended to


def test_main_emits_the_per_step_ledger_to_stderr_at_default_config(
    platform, wake_env, monkeypatch
):
    """The DoD (issue #248): a real wake's per-step ledger reaches stderr at default config —
    what was invisible in production because the wake CLI never configured logging."""
    monkeypatch.delenv("HARNESS_LOG_LEVEL", raising=False)
    _serve_openai_and_messages(platform, page(message(uuid=M0, body="status?")))

    stderr = io.StringIO()
    with _isolated_root_logging(), contextlib.redirect_stderr(stderr):
        assert main(["--timeline", TIMELINE_UUID]) == 0

    captured = stderr.getvalue()
    assert "wake used" in captured  # the reserve/step-count ledger line reached stderr
    assert re.search(r"step \d+/\d+", captured)  # …and the per-step ledger lines


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
    # The platform SDK (issue #303): the harness's one hard dependency, and the third
    # version-bearing one — cross-checked against the module's own `__version__`, an independent
    # source from the installed metadata the field reads.
    import basecradle

    assert report["platform_sdk_version"] == basecradle.__version__
    # The resolved active tool set — the platform tools that actually activate under the locked
    # policy, not a shipped-default list. The benign platform reads/writes are always present.
    assert {"memory", "users", "messages", "timelines", "lock", "delete"} <= set(report["tools"])
    # Powerful tools are opt-in (issue #168), so none activate in a default config.
    assert "generate_image" not in report["tools"]
    assert "web_search" not in report["builtins"]
    # And the opt-in manifest (issue #181) is empty for that safe default — nothing opted in.
    assert report["opt_in_tools"] == []
    # The MCP manifest (issue #261) is empty too — the default `mcp/` dir ships empty.
    assert report["mcp_servers"] == []
    # The resolved per-request MCP timeout (issue #320): unset env → the 20.0-second default, so
    # the NOC can audit an `mcp_timeout` axis off-box. A number, never null.
    assert report["mcp_request_timeout"] == 20.0
    # The memory axis (issue #269): the *bound* store, and the version of the package backing it
    # (none for the built-in sqlite store — it ships inside the harness).
    assert report["memory_provider"] == "sqlite"
    assert report["memory_provider_version"] is None
    # Model params (issue #236): absent file → an empty object and no collisions, present in the
    # additive contract so a verifier can rely on the keys existing.
    assert report["model_params"] == {}
    assert report["model_params_stripped"] == []


def test_resolved_config_reports_loaded_model_params_and_collisions(
    wake_env, monkeypatch, tmp_path
):
    """`--resolved-config` emits the operator's `model_params.json` verbatim plus the keys the
    active SDK's build drops as harness-owned collisions (issue #236) — the wire-level proof a
    tuning like `reasoning: {effort: high}` is loaded, which no other field ever showed. The
    default @jt stack is the `openai` SDK, so `model` collides (identity is AI_MODEL) while
    `reasoning` passes through untouched."""
    cfg = tmp_path / "cfg"
    monkeypatch.setenv("BASECRADLE_CONFIG_HOME", str(cfg))
    cfg.mkdir()
    (cfg / "model_params.json").write_text(
        json.dumps({"reasoning": {"effort": "high"}, "temperature": 0.2, "model": "sneaky"}),
        encoding="utf-8",
    )

    report = resolved_config()

    # Verbatim: exactly what the operator wrote, harness-owned key included (reporting is honest;
    # the build strips it at call time).
    assert report["model_params"] == {
        "reasoning": {"effort": "high"},
        "temperature": 0.2,
        "model": "sneaky",
    }
    # Only `model` collides on the openai SDK; the genuine tuning keys reach the call.
    assert report["model_params_stripped"] == ["model"]


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


def test_resolved_config_reports_configured_mcp_servers_regardless_of_load(
    wake_env, monkeypatch, tmp_path
):
    """The MCP manifest (issue #261) the NOC's MCP-overlay drift audit keys on: the sorted
    **names** of the configured `mcp/<name>.json` drop-ins, reported from the on-disk config
    (`load_mcp_configs`) — so a server that fails to connect this run still appears, and a
    transient upstream blip never reads as desired-state drift. The default empty dir → `[]`."""
    cfg = tmp_path / "cfg"
    monkeypatch.setenv("BASECRADLE_CONFIG_HOME", str(cfg))
    cfg.mkdir()
    assert resolved_config()["mcp_servers"] == []  # no mcp/ dir → empty manifest

    # Two configured servers, one of which cannot possibly connect (a bogus stdio command that
    # fails fast). Both are *configured* on disk, so both name the manifest — independent of load.
    mcp_dir = cfg / "mcp"
    mcp_dir.mkdir()
    (mcp_dir / "workmail.json").write_text(
        json.dumps({"command": "/nonexistent-mcp-binary"}), encoding="utf-8"
    )
    (mcp_dir / "notes.json").write_text(
        json.dumps({"command": "/also-nonexistent"}), encoding="utf-8"
    )

    report = resolved_config()

    # Sorted configured names — the ground-truth signal — regardless of connect success.
    assert report["mcp_servers"] == ["notes", "workmail"]
    # And the failed load is honestly recorded in `skipped` (the loaded set diverges from the
    # configured set) — proving `mcp_servers` reports configuration, not liveness.
    assert "workmail" in report["skipped"]


def test_resolved_config_reports_the_resolved_mcp_request_timeout(wake_env, monkeypatch):
    """The MCP-timeout axis (issue #320): `--resolved-config` emits the **resolved** per-request
    MCP timeout — the exact value a wake would use, not a re-read of the raw env — so the NOC can
    add an audited `mcp_timeout` `agent.env` axis and confirm off-box (their #195 law) that a
    browser-using agent got the longer navigation headroom it needs."""
    # Unset → the 20.0-second default. Assert the value, not merely presence: a present-but-wrong
    # timeout would pass a presence check while an agent silently stalls or over-waits.
    monkeypatch.delenv("HARNESS_MCP_TIMEOUT", raising=False)
    assert resolved_config()["mcp_request_timeout"] == 20.0

    # A positive override is resolved through the same `_timeout_from_env` path the wake uses — a
    # browser agent's 60-second navigation headroom shows through, so the axis is verifiable.
    monkeypatch.setenv("HARNESS_MCP_TIMEOUT", "60")
    assert resolved_config()["mcp_request_timeout"] == 60.0

    # A non-positive or malformed value is not ground truth for what the wake does — it falls back
    # to the default, exactly as `load_mcp_tools` would, so the reported number never lies.
    monkeypatch.setenv("HARNESS_MCP_TIMEOUT", "0")
    assert resolved_config()["mcp_request_timeout"] == 20.0
    monkeypatch.setenv("HARNESS_MCP_TIMEOUT", "not-a-number")
    assert resolved_config()["mcp_request_timeout"] == 20.0


def test_resolved_config_reports_the_default_memory_provider_so_a_silent_fallback_is_visible(
    wake_env, monkeypatch
):
    """The memory axis (issue #269). With no `HARNESS_MEMORY_PROVIDER` the harness binds the
    default SQLite store — and *says so*. That is the whole point: before this field, an agent
    that lost the var from its `agent.env` fell back to SQLite, quietly abandoned its palace, and
    stayed byte-indistinguishable from a MemPalace agent in every off-box signal. `sqlite` carries
    no version — its store ships inside the harness, whose version is `harness_version`."""
    monkeypatch.delenv("HARNESS_MEMORY_PROVIDER", raising=False)

    report = resolved_config()

    assert report["memory_provider"] == "sqlite"
    assert report["memory_provider_version"] is None


def test_resolved_config_reports_a_bound_mempalace_provider(wake_env, monkeypatch):
    """Env var set → the *bound* provider is reported, read off the object `_resolve_tools`
    actually built (not a re-read of the var — the `--resolved-config` env-gap class,
    basecradle-noc#62). The version is `None` here because MemPalace is an optional extra this
    suite never installs — which is exactly the off-box defect signal it exists to give: bound to
    a palace whose package is absent (binding is lazy), so the agent loses its memory at the first
    wake while every other signal reads green."""
    monkeypatch.setenv("HARNESS_MEMORY_PROVIDER", "mempalace")

    report = resolved_config()

    assert report["memory_provider"] == "mempalace"
    assert report["memory_provider_version"] is None
    # The provider's own tools fold into the resolved set, so a MemPalace agent carries its
    # read-only `memory_search` (issue #267) and *not* the SQLite `memory` tool. The tool axis
    # is still no substitute for the field, though: a provider may contribute no tool at all, so
    # only `memory_provider` names the bound store.
    assert "memory_search" in report["tools"]
    assert "memory" not in report["tools"]


def test_resolved_config_reports_the_installed_platform_sdk_not_the_declared_pin(wake_env):
    """The platform-SDK axis (issue #303). The version is read from the *installed* distribution —
    what this venv can actually import — never from `pyproject`'s `basecradle>=0.6` pin, which is
    what the harness declares about itself and would read green on a venv that never got the
    upgrade. That gap is the whole defect: a `--resolved-config` builds no platform client, so an
    agent sitting on `basecradle` 0.5.x passed every drift axis and `TypeError`d on
    `idempotency_key=` the first time it spoke."""
    import basecradle

    report = resolved_config()

    assert report["platform_sdk_version"] == basecradle.__version__
    assert report["platform_sdk_version"] == metadata.version("basecradle")
    # Never an empty string: `""` is falsy, prints as nothing, and reads as "fine" to a drift
    # check written against a truthy value. Absent → `None` (the test below), present → a version.
    assert report["platform_sdk_version"] != ""


def test_resolved_config_reports_a_missing_platform_sdk_as_null_never_empty(wake_env, monkeypatch):
    """An uninstalled platform SDK is `None` — a defect signal, not a shrug (issue #303): the
    harness's one hard dependency is gone, so the agent has no way to reach the platform at all.
    It must not degrade into `""`, which a drift check reading truthiness would pass. (The
    consumer's other half: a harness too old to *have* the field is a **missing key**, which the
    NOC treats as an error rather than a silent skip — the two states stay distinguishable.)"""

    def missing(dist: str) -> str:
        if dist == "basecradle":
            raise metadata.PackageNotFoundError(dist)
        return metadata.version(dist)

    monkeypatch.setattr(
        _wake,
        "metadata",
        SimpleNamespace(version=missing, PackageNotFoundError=metadata.PackageNotFoundError),
    )

    report = resolved_config()

    assert report["platform_sdk_version"] is None
    assert "platform_sdk_version" in report  # present-and-null ≠ absent; the NOC reads both


def test_resolved_config_reports_the_active_profile_locked_by_default(wake_env):
    """The `active_profile` field (issue #256): a default deploy sets no HARNESS_PROFILE, so it
    reports `locked` — the ground truth a shell-class enablement's live-verify reads."""
    assert resolved_config()["active_profile"] == "locked"


def test_resolved_config_unlocked_lists_an_opted_in_shell_under_tools(wake_env, monkeypatch):
    """Under `HARNESS_PROFILE=unlocked` the opted-in shell resolves into the active `tools` set
    (not `skipped`) and `active_profile` reads `unlocked`, so `--resolved-config` can confirm a
    shell-class enablement actually landed — unverifiable before this (issue #256)."""
    monkeypatch.setattr(os, "geteuid", lambda: 1000, raising=False)  # deterministic non-root
    install(os.environ["BASECRADLE_CONFIG_HOME"], provider="openai", opt_in=["shell"])
    monkeypatch.setenv("HARNESS_PROFILE", "unlocked")

    report = resolved_config()

    assert report["active_profile"] == "unlocked"
    assert "shell" in report["tools"]
    assert "shell" not in report["skipped"]
    assert "shell" in report["opt_in_tools"]  # still an opt-in (powerful) tool


def test_resolved_config_locked_skips_an_opted_in_shell(wake_env, monkeypatch):
    """The safe default is unchanged: with the same opted-in shell but no HARNESS_PROFILE (locked),
    the policy filters shell to `skipped` and it never reaches the active `tools` set."""
    monkeypatch.setattr(os, "geteuid", lambda: 1000, raising=False)
    install(os.environ["BASECRADLE_CONFIG_HOME"], provider="openai", opt_in=["shell"])
    monkeypatch.delenv("HARNESS_PROFILE", raising=False)

    report = resolved_config()

    assert report["active_profile"] == "locked"
    assert "shell" not in report["tools"]
    assert "shell" in report["skipped"]


def test_main_wake_admits_an_opted_in_shell_under_the_unlocked_profile(
    platform, wake_env, monkeypatch
):
    """The wake path builds the registry from the deploy-selected profile (issue #256): with the
    shell opted in AND `HARNESS_PROFILE=unlocked`, the unlocked registry admits it and the wake
    runs clean. Were the profile threaded into resolution but not into `Harness(policy=…)`, the
    kept-but-forbidden shell would raise `PolicyError` at registration and this would exit non-zero.
    """
    monkeypatch.setattr(os, "geteuid", lambda: 1000, raising=False)
    install(os.environ["BASECRADLE_CONFIG_HOME"], provider="openai", opt_in=["shell"])
    monkeypatch.setenv("HARNESS_PROFILE", "unlocked")
    _serve_openai_and_messages(platform, page(message(uuid=M0, body="run something")))

    assert main(["--timeline", TIMELINE_UUID]) == 0
    assert platform.post(f"/timelines/{TIMELINE_UUID}/messages").called  # the unlocked wake replied


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


def test_the_brief_is_shown_on_every_wake(platform, tmp_path):
    """Turn 0 is persistent: the model is shown a fresh brief on every wake, not just turn 1.

    Two wakes over one home (two router-spawned processes). Each engages the model, so each
    composes and shows a fresh brief — standing context that is always *recent* in the
    conversation rather than aging out at the top.
    """
    serve_dashboard_md(platform)
    manifest = [("memory", None), ("lock", "one-way and irreversible.")]

    serve_messages(platform, page(message(uuid=M0, body="What's the status?")))
    agent1, model1 = build_wake(tmp_path, onboard=True, tool_manifest=manifest)
    agent1.wake()
    assert len(_brief_shown(model1)) == 1

    # A fresh process: the mark is now M0, so wake 2 replies only to the newer M1.
    serve_messages(
        platform, page(message(uuid=M1, body="any update?"), message(uuid=M0, body="hi"))
    )
    agent2, model2 = build_wake(tmp_path, onboard=True, tool_manifest=manifest)
    agent2.wake()

    assert len(_brief_shown(model2)) == 1  # wake 2's own brief — exactly one, never a stack


def test_a_wake_persists_no_brief(platform, tmp_path):
    """ISSUE #275 — the whole point: the brief is ephemeral, and the transcript proves it.

    The brief is recomposed every wake (current time, step budget, live dashboard), so a persisted
    copy is a stale duplicate the model re-reads and re-pays for on every later turn — 47% of one
    agent's 754 K-token context was ~66 such copies. Two wakes here: the model sees the brief both
    times, and the stored transcript carries not one byte of it, across processes.
    """
    serve_dashboard_md(platform)
    serve_messages(platform, page(message(uuid=M0, body="What's the status?")))
    agent1, model1 = build_wake(tmp_path, onboard=True, tool_manifest=[("memory", None)])
    agent1.wake()

    serve_messages(
        platform, page(message(uuid=M1, body="any update?"), message(uuid=M0, body="hi"))
    )
    agent2, model2 = build_wake(tmp_path, onboard=True, tool_manifest=[("memory", None)])
    agent2.wake()

    assert _brief_shown(model1) and _brief_shown(model2)  # both wakes showed the model a brief…
    assert _brief_turns(agent2) == []  # …and neither wrote one to the transcript
    # Belt and braces: nothing brief-shaped survives on disk either (a fresh reader sees none).
    stored = json.loads((tmp_path / "sessions" / f"timeline%3A{TIMELINE_UUID}.json").read_text())
    assert not [t for t in stored if "How to operate here" in (t.get("content") or "")]


def test_a_failed_wake_persists_no_brief(platform, tmp_path):
    """A wake whose model call *fails* must not grow the transcript by a brief either.

    The pre-#275 bug wrote the brief before `engine.run`, so even an erroring wake — one that
    accomplished nothing — added ~20 KB to every future wake's bill. Now the brief only ever
    rides the message list handed to the provider, so a failure leaves the partial transcript
    (issue #244's ledger) without it.
    """

    class Boom:
        provider, model = "openai", "gpt-4o"

        def chat(self, messages, tools=None):
            raise RuntimeError("model exploded mid-wake")

    serve_dashboard_md(platform)
    serve_messages(platform, page(message(uuid=M0, body="hi")))
    agent, _ = build_wake(tmp_path, provider=Boom(), onboard=True, tool_manifest=[("memory", None)])

    with pytest.raises(RuntimeError):
        agent.wake()

    assert _brief_turns(agent) == []  # no brief…
    history = agent.harness.session(agent.source).history
    assert any("turn failed" in (m.content or "") for m in history)  # …but the ledger is intact


def test_the_brief_composes_all_four_parts(platform, tmp_path):
    """The brief is initialize.md + the live tool manifest + the live dashboard + personality."""
    serve_dashboard_md(platform, text="# Live Dashboard\n\nWho you are, where everything is.\n")
    serve_messages(platform, page(message(uuid=M0, body="hi")))
    agent, model = build_wake(
        tmp_path,
        onboard=True,
        tool_manifest=[("memory", None), ("lock", "one-way and irreversible.")],
    )

    agent.wake()

    brief = _brief_shown(model)[0].content
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
    agent, model = build_wake(tmp_path, onboard=True, tool_manifest=[("memory", None)])

    posted = agent.wake()

    assert len(posted) == 1  # the wake still replied — the fetch failure never broke it
    brief = _brief_shown(model)[0].content
    assert "How to operate here" in brief  # initialize.md present…
    assert "Your active tools right now:" in brief  # …and the manifest…
    assert "You are a helpful peer on BaseCradle." in brief  # …and the personality, sans dashboard


def test_onboarding_off_shows_no_brief(platform, tmp_path):
    """`onboard=False` wakes with only the operator's charter — no persistent brief."""
    serve_messages(platform, page(message(uuid=M0, body="hi")))
    agent, model = build_wake(tmp_path, onboard=False)

    agent.wake()

    assert _brief_shown(model) == []


def test_an_idle_wake_shows_no_brief_and_skips_the_dashboard_fetch(platform, tmp_path):
    """Nothing unseen → the model is never engaged → no brief, and no live dashboard fetch.

    The lazy, once-per-wake composition means an idle (or probe-only) wake pays nothing: no
    model call, and no live dashboard fetch to compose a brief nobody will read.
    """
    route = serve_dashboard_md(platform)
    MarkStore(tmp_path).set(TIMELINE_UUID, M0)  # caught up → nothing new this wake
    serve_messages(platform, page(message(uuid=M0, body="hi")))
    agent, provider = build_wake(tmp_path, onboard=True, tool_manifest=[("memory", None)])

    posted = agent.wake()

    assert posted == []
    assert provider.prompts == []  # the model was never engaged
    assert _brief_shown(provider) == []
    assert not route.called  # lazy: no engagement → the live dashboard was never fetched


def test_the_brief_precedes_the_item_it_governs(platform, tmp_path):
    """The brief sits just *ahead* of the user turn it governs, at the tail of the message list.

    Position is load-bearing twice over. Ahead of the item, so it contextualizes the thing being
    answered. And at the **tail** — after the whole frozen transcript — because provider prefix
    caching only pays out on a byte-stable prefix: moving this volatile block to position 0
    ("system prompts go first") would change the prefix on every request and silently destroy
    caching fleet-wide. Stable content first, volatile content last (issue #275).
    """
    serve_dashboard_md(platform)
    serve_messages(platform, page(message(uuid=M0, body="What's the status?")))
    # A silent brain: one provider call, so `last_messages` *is* the turn's opening call, which is
    # where the brief's position is decided. (A speaking turn's last call trails a tool result.)
    agent, model = build_wake(
        tmp_path,
        CountingProvider(speak=False),
        onboard=True,
        tool_manifest=[("memory", None)],
    )

    agent.wake()

    shown = model.last_messages
    brief_idx = next(i for i, m in enumerate(shown) if _is_brief(m))
    user_idx = next(i for i, m in enumerate(shown) if m.role == "user")
    assert brief_idx == user_idx - 1  # …immediately ahead of the message it contextualizes…
    # …and nothing volatile sits ahead of it: the brief and the turn it governs are the tail,
    # trailed only by the engine's step-counter note. Everything before is frozen transcript.
    assert [m.role for m in shown[brief_idx:]] == ["system", "user", "system"]
    assert "assistant" in [m.role for m in agent.harness.session(agent.source).history]


def test_a_brief_composition_failure_does_not_break_the_wake(platform, tmp_path, monkeypatch):
    """A raise inside brief composition (e.g. an IO error reading a prompt file) degrades to
    no brief — the wake still replies, never crashes. Same invariant the dashboard fetch holds."""
    import basecradle_harness._wake as wake_mod

    def boom(*args, **kwargs):
        raise OSError("permission denied reading prompts/initialize.md")

    monkeypatch.setattr(wake_mod, "prompt_text", boom)
    serve_dashboard_md(platform)
    serve_messages(platform, page(message(uuid=M0, body="hi")))
    agent, model = build_wake(tmp_path, onboard=True, tool_manifest=[("memory", None)])

    posted = agent.wake()

    assert len(posted) == 1  # the wake still replied despite the compose failure
    assert _brief_shown(model) == []  # …with no brief, rather than crashing


class _DumpingTool(Tool):
    """A tool whose result is huge — the mailbox listing that drove issue #275 (142 KB, live)."""

    name = "mailbox"
    description = "List the mailbox."
    DUMP = "MAILBOX\n" + ("x" * 100_000) + "\nEND-OF-MAILBOX"

    def run(self, **kwargs) -> str:
        return self.DUMP


class _ToolThenText:
    """Calls the tool on its first turn of a wake, then settles to text."""

    provider, model = "openai", "gpt-4o"

    def __init__(self) -> None:
        self.read: list[list[tuple[str, str | None]]] = []  # (role, content) per call, at call time

    def chat(self, messages, tools=None):
        self.read.append([(m.role, m.content) for m in messages])
        if not any(m.role == "tool" for m in messages):
            return Message.assistant(tool_calls=[ToolCall(id="c1", name="mailbox", arguments={})])
        return Message.assistant(content="You have mail.")


def test_a_huge_tool_result_does_not_tax_the_next_wake(platform, tmp_path):
    """ISSUE #275, end to end on the deployed path: the wake that ran the tool reads it whole;
    the wake *after* it pays a bounded price.

    This is the failure this issue exists to end — 39% of one agent's 754 K-token context was raw
    tool output, three mailbox dumps of 142/120/101 KB among it, re-sent on every wake for the life
    of the timeline. Two router-spawned processes over one home: wake 1 runs the tool, wake 2
    inherits the transcript.
    """
    serve_dashboard_md(platform)
    serve_messages(platform, page(message(uuid=M0, body="check my mail")))
    model1 = _ToolThenText()
    harness1 = Harness(model1, tools=[_DumpingTool()], home=tmp_path)
    agent1 = WakeAgent(harness1, timeline=TIMELINE_UUID, client=BaseCradle(token=FAKE_TOKEN))
    agent1.wake()

    # Wake 1 read the dump in full — the model is never handed a truncated tool result on the
    # turn it asked for it.
    assert ("tool", _DumpingTool.DUMP) in model1.read[1]

    # A fresh process over the same home: the transcript it inherits is bounded.
    serve_messages(
        platform, page(message(uuid=M1, body="anything else?"), message(uuid=M0, body="hi"))
    )
    model2 = _ToolThenText()
    harness2 = Harness(model2, tools=[_DumpingTool()], home=tmp_path)
    agent2 = WakeAgent(harness2, timeline=TIMELINE_UUID, client=BaseCradle(token=FAKE_TOKEN))
    agent2.wake()

    inherited = next(c for role, c in model2.read[0] if role == "tool")
    assert len(inherited) < 5_000  # the dump was 100 KB; the next wake re-reads ~2.5 KB of it
    assert "elided" in inherited and f"of {len(_DumpingTool.DUMP)}" in inherited
    # …and the transcript is still well-formed: the elided result still answers its call, so the
    # assistant tool-call turn is not left dangling (which would break every wake after it).
    stored = json.loads((tmp_path / "sessions" / f"timeline%3A{TIMELINE_UUID}.json").read_text())
    call_ids = [c["id"] for t in stored for c in t.get("tool_calls", [])]
    assert call_ids and call_ids == [t["tool_call_id"] for t in stored if t["role"] == "tool"]


def test_the_brief_is_composed_once_per_wake_across_many_items(platform, tmp_path):
    """A multi-item wake composes the brief once (one dashboard fetch) and shows it every call.

    Two model calls this wake (a message batch, then an activated task). Each is shown the brief
    ahead of its own user turn — standing context is never stale by an item — yet the live
    dashboard is fetched exactly once, and nothing lands in the transcript.
    """
    route = serve_dashboard_md(platform)
    serve_messages(platform, page(message(uuid=M0, body="hi")))
    serve_tasks(platform, task_page(task(uuid=T0, instructions="Post the daily summary.")))
    agent, model = build_wake(tmp_path, onboard=True, tool_manifest=[("memory", None)])

    agent.wake()

    assert len(model.prompts) == 2  # the message batch, then the activated task
    assert len(_brief_shown(model)) == 1  # the task call, too, was shown the brief…
    assert route.call_count == 1  # …composed once — the live dashboard was fetched once
    assert _brief_turns(agent) == []  # …and neither call persisted it


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

    # **The alert is a log line, never a post** (issue #293). The harness does not speak for the
    # agent, and the breaker alert was the one message it still wrote in the agent's own voice. The
    # operator's breadcrumb is the WARNING (which is exactly what the NOC alerts on); the peers see
    # what actually happened — an agent that went quiet. So the *only* bodies on this timeline are
    # the two replies the agent itself chose to send, before the burst tripped it.
    assert _alert_bodies(platform) == ["Hello, John.", "Hello, John."]


def test_the_breaker_auto_resets_and_resumes_after_the_burst_clears(platform, tmp_path, caplog):
    """Once the burst clears past the cooldown, the next wake auto-resets: it logs the recovery
    alert and resumes normal operation (engages the message it had been declining)."""
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
    with caplog.at_level("WARNING", logger="basecradle_harness"):
        posted = wake_with(message(uuid=M3, body="three"), message(uuid=M2, body="two"))
    assert len(posted) == 1  # the agent spoke — through its tool, as it now always does
    assert provider.prompts == [
        "[2026-06-04T00:00:00.000Z] john: one",
        "[2026-06-04T00:00:00.000Z] john: two",
        "[2026-06-04T00:00:00.000Z] john: three",
    ]  # resumed
    assert MarkStore(tmp_path).get(TIMELINE_UUID) == M3
    # The recovery alert is a WARNING, not a post (issue #293). The only bodies on this timeline are
    # the three replies the agent itself chose to send — the two healthy wakes before the trip, and
    # this one, resumed. No alert, no harness-authored word in the agent's voice.
    resets = [r.getMessage() for r in caplog.records if "Wake breaker RESET" in r.getMessage()]
    assert len(resets) == 1
    assert _alert_bodies(platform) == ["Hello, John."] * 3


def test_a_tripped_wake_never_touches_the_timeline(platform, tmp_path):
    """A tripped wake is inert: no provider call, and **no post of any kind** (issue #293).

    This once tested that the breaker's alert post degraded gracefully on a locked timeline. There
    is no alert post any more — the breaker speaks to the operator, in the journal — so what is
    worth pinning is the stronger property that replaced it: a tripped wake writes *nothing* to the
    timeline, which is why a locked timeline can no longer refuse anything it does.
    """
    clock = FakeClock()
    provider = CountingProvider()
    # Pre-fill the window so the very next wake trips (cap 1; two recorded wakes already in window).
    pre = WakeBreaker(tmp_path, max_wakes=1, window=60.0, now=clock)
    pre.record_and_check(TIMELINE_UUID)
    pre.record_and_check(TIMELINE_UUID)  # now tripped on disk
    serve_messages(platform, page(message(uuid=M0, body="hi")))

    posted = _wake_with_breaker(tmp_path, provider, clock, max_wakes=1).wake()  # must not raise

    assert posted == []
    assert provider.prompts == []  # tripped → no provider call
    assert _alert_bodies(platform) == []  # …and nothing reached the timeline


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
    """A peer AI's message → sleep(max(floor, chars/rate) - age); here age 0, 510 chars → 30s."""
    pacer, sleep = _pacer()
    slept = pacer.pace(_pace_msg("x" * 510))  # 510 / 17 chars-per-sec = 30s, above the 20s floor

    assert slept == 30.0
    assert sleep.calls == [30.0]


def test_the_floor_applies_to_a_very_short_message():
    """A one-word peer-AI reply reads in a blink, but the 20s floor keeps it human-paced."""
    pacer, sleep = _pacer()
    slept = pacer.pace(_pace_msg("ok"))  # 2 / 17 = 0.1s → clamped up to the 20s floor

    assert slept == 20.0
    assert sleep.calls == [20.0]


def test_delay_scales_with_message_length():
    """Twice the characters → twice the read-time (above the floor), a true length scaling."""
    short, short_sleep = _pacer()
    long, long_sleep = _pacer()

    assert short.pace(_pace_msg("x" * 510)) == 30.0  # 510 / 17
    assert long.pace(_pace_msg("x" * 1020)) == 60.0  # 1020 / 17 — proportionally longer
    assert short_sleep.calls == [30.0]
    assert long_sleep.calls == [60.0]


def test_age_is_subtracted_so_only_the_remainder_is_waited():
    """LOAD-BEARING `- age`: a message already 5s old owes only the remaining 25s of its 30s read."""
    pacer, sleep = _pacer(now=PACE_CREATED.replace(second=5))  # message aged 5s since it appeared
    slept = pacer.pace(_pace_msg("x" * 510))  # target 30s, age 5s → wait 25s

    assert slept == 25.0
    assert sleep.calls == [25.0]


def test_a_message_older_than_its_read_time_is_not_paced():
    """When age >= target the remainder clamps to 0 — no sleep (the 'quicker across timelines' case)."""
    pacer, sleep = _pacer(now=PACE_CREATED.replace(minute=5))  # 300s old, far past a 20s read
    slept = pacer.pace(_pace_msg("x" * 400))

    assert slept == 0.0
    assert sleep.calls == []  # never slept


def test_a_negative_age_is_clamped_so_the_delay_never_exceeds_the_read_time():
    """LOAD-BEARING clamp: a future-dated stamp / lagging box clock must not inflate the sleep."""
    # The box clock is 5 minutes behind the message's `created_at` → age is -300s. Unclamped,
    # `target - age` would be 330s; clamped, the message owes only its full 30s read-time.
    pacer, sleep = _pacer(now=PACE_CREATED - timedelta(minutes=5))
    slept = pacer.pace(_pace_msg("x" * 510))

    assert slept == 30.0  # target, never target + skew
    assert sleep.calls == [30.0]


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

    assert pacer.pace(_pace_msg("x" * 400)) == 20.0  # falls back to the floor, no ZeroDivisionError
    assert sleep.calls == [20.0]


# --- pacing wired through a real wake (the reply path) ------------------------


def test_a_wake_paces_before_answering_a_peer_ai(platform, tmp_path):
    """End-to-end: a peer AI's message is read-paced, then answered exactly as today."""
    serve_messages(platform, page(peer_ai_message(uuid=M0, body="x" * 510)))
    pacer, sleep = _pacer()
    agent, provider = build_wake(tmp_path, pacer=pacer)

    posted = agent.wake()

    assert sleep.calls == [30.0]  # paced the peer AI's 510-char message (510 / 17)
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
    assert _pace_chars_per_sec_from_env() == 17.0  # the real default (issue #226)
    assert _pace_floor_seconds_from_env() == 20.0

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


# --- the settle loop + mid-generation staleness guard (issue #226) ------------
#
# The 0.44.0 pacer took a snapshot, slept, then replied to the snapshot — so a message that
# landed during the sleep (Loop 1) or during generation (Loop 2) made the reply stale. The
# rework closes both windows. These tests drive a SCRIPTABLE platform whose message list can
# change mid-wake: a `RecordingSleep` hook makes a message "arrive" during the read (Loop 1),
# and a provider hook makes one "arrive" during the model call (Loop 2). Deterministic — the
# clock and sleep are injected, so nothing actually waits.


class ScriptedMessages:
    """A mutable newest-first message list served at `/messages`, re-read live on every call.

    Every `/messages` read returns the *current* list, so a message inserted mid-wake (by a
    sleep hook in Loop 1, or a provider hook in Loop 2) is seen by the next re-read exactly as
    a real arrival would be. `arrive` prepends (newest-first); the harness's mark filters what
    is genuinely new.
    """

    def __init__(self, platform, *initial):
        self._messages = list(initial)
        platform.get("/messages").mock(
            side_effect=lambda request: httpx.Response(200, json=page(*self._messages))
        )

    def arrive(self, wire_message):
        self._messages.insert(0, wire_message)  # newest-first


class HookedProvider:
    """A canned brain that runs `on_chat(call_index)` before each reply — the Loop-2 seam.

    The hook lets a test make a message "arrive" during a specific generation (e.g. only the
    first), so the post-generation staleness re-check sees it and rebuilds. Records every
    prompt so build count and batch contents are assertable.
    """

    def __init__(self, text="Hello, John.", on_chat=None):
        self.text = text
        self.on_chat = on_chat
        self.prompts: list[str] = []

    def chat(self, messages, tools=None):
        # Record the last real turn's text, skipping the engine's step-counter note (issue #243).
        self.prompts.append(_convo(messages)[-1].content)
        if self.on_chat is not None:
            self.on_chat(len(self.prompts))
        return Message.assistant(content=self.text)


# --- Loop 1: the settle loop --------------------------------------------------


def test_a_newer_ai_message_during_the_read_restarts_the_settle(platform, tmp_path):
    """A newer peer-AI message landing during the read-pace folds in and restarts the wait.

    The 0.44.0 doublet defect: pace message N, and while sleeping message N+1 lands → a
    separate wake replies to N+1, one turn behind. The settle loop re-reads after the sleep;
    a newer peer AI restarts the pace on it, so this one wake reacts to the settled newest.
    """
    scripted = ScriptedMessages(platform, peer_ai_message(uuid=M0, body="first from Brain"))
    sleep = RecordingSleep()

    # Brain's second message lands during the FIRST read-pace; the second read-pace is quiet.
    def recording_then_arrive(seconds):
        sleep(seconds)
        if len(sleep.calls) == 1:
            scripted.arrive(peer_ai_message(uuid=M1, body="and a follow-up from Brain"))

    pacer = ReadPacer(clock=lambda: PACE_CREATED, sleep=recording_then_arrive)
    agent, provider = build_wake(tmp_path, HookedProvider(), pacer=pacer)

    posted = agent.wake()

    assert len(sleep.calls) == 2  # paced M0, a newer AI landed → restarted, paced M1, then settled
    # ONE batched turn covering BOTH of Brain's messages — not a doublet. The build count is what
    # says so now: this brain is silent (it calls no tool), so nothing reaches the timeline (#293).
    assert len(provider.prompts) == 1
    assert posted == []
    assert provider.prompts == [
        "[2026-06-04T00:00:00.000Z] briggs: first from Brain\n"
        "[2026-06-04T00:00:00.000Z] briggs: and a follow-up from Brain"
    ]
    assert MarkStore(tmp_path).get(TIMELINE_UUID) == M1  # mark past the newest arrival


def test_a_human_arriving_during_the_read_settles_immediately(platform, tmp_path):
    """A human message landing during the read-pace ends the settle at once — respond now.

    The settle only restarts for a newer *AI*; a human arrival breaks the loop (folded into the
    batch), so a human "STOP!" during an AI read is answered without another read-delay.
    """
    scripted = ScriptedMessages(platform, peer_ai_message(uuid=M0, body="a long AI message"))
    sleep = RecordingSleep()

    def recording_then_arrive(seconds):
        sleep(seconds)
        if len(sleep.calls) == 1:
            scripted.arrive(message(uuid=M1, body="STOP!"))  # john, human

    pacer = ReadPacer(clock=lambda: PACE_CREATED, sleep=recording_then_arrive)
    agent, provider = build_wake(tmp_path, HookedProvider(), pacer=pacer)

    posted = agent.wake()

    assert len(sleep.calls) == 1  # the human arrival settled it — no second read-pace
    assert len(provider.prompts) == 1  # one batched turn over both messages
    assert posted == []  # a silent brain: it thought, and said nothing
    assert provider.prompts == [
        "[2026-06-04T00:00:00.000Z] briggs: a long AI message\n"
        "[2026-06-04T00:00:00.000Z] john: STOP!"
    ]


# --- Loop 2: the mid-generation staleness guard -------------------------------


def test_a_message_arriving_during_generation_triggers_a_rebuild(platform, tmp_path):
    """A message landing *during* the model call folds into the batch and the reply regenerates.

    The generation window: the LLM call itself takes seconds, and a message that lands while it
    runs would otherwise be answered-over. Loop 2 re-reads after generating; a fresh message
    triggers a rebuild that seeds it before the reply posts.
    """
    scripted = ScriptedMessages(platform, message(uuid=M0, body="original question"))

    def on_chat(call_index):
        if call_index == 1:  # a message lands during the first generation
            scripted.arrive(message(uuid=M1, body="wait, also this"))

    agent, provider = build_wake(tmp_path, HookedProvider(on_chat=on_chat))

    posted = agent.wake()

    assert len(provider.prompts) == 2  # generated once, saw the new message, rebuilt once
    assert provider.prompts[1] == (
        "[2026-06-04T00:00:00.000Z] john: original question\n"
        "[2026-06-04T00:00:00.000Z] john: wait, also this"
    )  # the rebuild folded the mid-generation arrival in
    assert posted == []  # the rebuilt turn is silent; only the batch it saw was under test
    assert MarkStore(tmp_path).get(TIMELINE_UUID) == M1


def test_no_mid_generation_arrival_is_a_single_build(platform, tmp_path):
    """The steady case: nothing arrives during generation → one build, one post, no rebuild."""
    ScriptedMessages(platform, message(uuid=M0, body="just this"))
    agent, provider = build_wake(tmp_path, HookedProvider())

    posted = agent.wake()

    assert len(provider.prompts) == 1  # no rebuild
    assert posted == []  # silent brain


def test_the_max_builds_cap_stands_unconditionally_and_leaves_the_last_arrival_unseen(
    platform, tmp_path
):
    """A message on *every* build would rebuild forever; `MAX_BUILDS` caps it and the build stands.

    The Nth build stands with no staleness check after it, so the burst can't stall the turn. The
    message that lands during that final build is left **unseen** (mark not advanced past it, not
    claimed), so it drives the next wake rather than being lost.
    """
    scripted = ScriptedMessages(platform, message(uuid=M0, body="q0"))
    arrivals = [M1, M2, M3]

    def on_chat(call_index):
        # A new message lands during every generation, so staleness never clears on its own.
        scripted.arrive(message(uuid=arrivals[call_index - 1], body=f"q{call_index}"))

    agent, provider = build_wake(tmp_path, HookedProvider(on_chat=on_chat))  # default max_builds=3

    posted = agent.wake()

    assert len(provider.prompts) == 3  # capped at MAX_BUILDS, not spinning
    assert posted == []  # the 3rd build stands unconditionally (this brain says nothing)
    # M1 and M2 (arrived during builds 1 and 2) were folded and marked; M3 (during build 3)
    # was left unseen for the next wake.
    assert MarkStore(tmp_path).get(TIMELINE_UUID) == M2
    assert ClaimStore(tmp_path).claim(TIMELINE_UUID, M3, kind="messages") is True  # M3 unclaimed


def test_a_human_message_during_generation_triggers_a_rebuild_all_senders(platform, tmp_path):
    """Loop 2 counts ALL senders: a human interjection mid-generation rebuilds too (not AI-only).

    Loop 1 is AI-only (watchability); Loop 2 is correctness, so a human message landing during
    generation must be seen before the agent answers — the "human posts STOP! mid-reply" case.
    """
    scripted = ScriptedMessages(platform, message(uuid=M0, body="a human question"))  # john

    def on_chat(call_index):
        if call_index == 1:
            scripted.arrive(message(uuid=M1, body="actually, never mind"))  # another human

    agent, provider = build_wake(tmp_path, HookedProvider(on_chat=on_chat))

    agent.wake()

    assert len(provider.prompts) == 2  # the human arrival triggered a rebuild
    assert "actually, never mind" in provider.prompts[1]


def test_disabling_pacing_skips_both_loops(platform, tmp_path):
    """`HARNESS_PACE_ENABLED=false` disables all of it: no read-pace, no staleness rebuild.

    The batch reply (the substrate) still stands, but Loop 1 never sleeps and Loop 2 does a
    single build even if a message lands during generation.
    """
    scripted = ScriptedMessages(platform, peer_ai_message(uuid=M0, body="x" * 510))

    def on_chat(call_index):
        scripted.arrive(peer_ai_message(uuid=M1, body="landed during generation"))

    sleep = RecordingSleep()
    disabled = ReadPacer(enabled=False, clock=lambda: PACE_CREATED, sleep=sleep)
    agent, provider = build_wake(tmp_path, HookedProvider(on_chat=on_chat), pacer=disabled)

    posted = agent.wake()

    assert sleep.calls == []  # Loop 1 skipped — no read-pace
    assert len(provider.prompts) == 1  # Loop 2 collapsed to a single build — no rebuild
    assert posted == []  # a silent brain
    # The message that arrived during the one build is left unseen for the next wake.
    assert ClaimStore(tmp_path).claim(TIMELINE_UUID, M1, kind="messages") is True


def test_max_builds_of_one_never_rebuilds(platform, tmp_path):
    """`max_builds=1` collapses Loop 2 to the pre-#226 single-shot even with pacing enabled."""
    scripted = ScriptedMessages(platform, message(uuid=M0, body="q0"))

    def on_chat(call_index):
        scripted.arrive(message(uuid=M1, body="q1"))

    agent, provider = build_wake(tmp_path, HookedProvider(on_chat=on_chat), max_builds=1)

    posted = agent.wake()

    assert len(provider.prompts) == 1  # one build, standing unconditionally
    assert posted == []  # a silent brain (a build that *spoke* is never rebuilt — see below)


# --- review hardening: tool side effects, probe acks, settle cap, orphan claims ---


class _ArrivingTool(Tool):
    """A tool that makes a message 'arrive' when it runs — to prove a tool-using build is not
    rolled back and re-fired by the Loop-2 staleness rebuild."""

    name = "poke"
    description = "A no-op tool used only to prove a tool-using build isn't rolled back."

    def __init__(self, scripted, wire_message):
        self.scripted = scripted
        self.wire_message = wire_message
        self.runs = 0

    def run(self, **kwargs):
        self.runs += 1
        self.scripted.arrive(self.wire_message)  # a message lands during the tool-using build
        return "poked"


class _SpeakingArrivalTool(MessagesTool):
    """The real `messages` tool — but a peer message lands *while the post is in flight*.

    The sharpest form of the Loop-2 race now that speech is a tool call (issue #293): the agent
    speaks, and a new message arrives during the very build that spoke. Rebuilding that build would
    post the same body a second time, so it must not be rebuilt — this tool is how the test forces
    the arrival to happen at exactly that instant.
    """

    def __init__(self, scripted, wire_message):
        super().__init__()
        self.scripted = scripted
        self.wire_message = wire_message
        self.runs = 0

    def run(self, **kwargs):
        self.runs += 1
        result = super().run(**kwargs)
        self.scripted.arrive(self.wire_message)  # a message lands during the speaking build
        return result


class _ToolThenReplyProvider:
    """First chat of a build → a tool call; the next → the final text. So one build runs a tool."""

    def __init__(self, tool_name, arguments=None):
        self.tool_name = tool_name
        self.arguments = (
            arguments
            if arguments is not None
            else ({"action": "create", "body": "Hello, John."} if tool_name == "messages" else {})
        )
        self.chats = 0

    def chat(self, messages, tools=None):
        self.chats += 1
        if self.chats == 1:
            return Message.assistant(
                tool_calls=[ToolCall(id="c1", name=self.tool_name, arguments=self.arguments)]
            )
        return Message.assistant(content="done")


def test_a_tool_using_build_is_not_rolled_back_when_a_message_arrives(platform, tmp_path):
    """A build that ran a tool has committed irreversible side effects → it is posted, never rebuilt.

    The Loop-2 rollback erases only the transcript, not a tool's real effects (a posted image, a
    sent message). So a build whose span contains a tool turn must NOT be rolled back and
    regenerated — otherwise the tool fires twice for one request. Here the tool itself makes a
    message arrive mid-build; the wake must still run it exactly once.
    """
    scripted = ScriptedMessages(platform, message(uuid=M0, body="do a thing"))
    tool = _ArrivingTool(scripted, message(uuid=M1, body="landed during the tool call"))
    client = BaseCradle(token=FAKE_TOKEN)
    harness = Harness(_ToolThenReplyProvider("poke"), tools=[tool], home=tmp_path)
    agent = WakeAgent(harness, timeline=TIMELINE_UUID, client=client, onboard=False)

    posted = agent.wake()

    assert tool.runs == 1  # the tool fired exactly once — no rollback+rebuild re-fired it
    assert posted == []  # `poke` is not a message tool, so the agent said nothing
    # M1 arrived during the tool-using build → left unseen for the next wake (not folded/rebuilt).
    assert ClaimStore(tmp_path).claim(TIMELINE_UUID, M1, kind="messages") is True


def test_a_build_that_spoke_is_never_rebuilt_so_the_agent_never_speaks_twice(platform, tmp_path):
    """The same guard, now aimed at the thing that matters most: **speech is a side effect** (#293).

    Since the agent speaks by calling a tool, a build that posted a message is a build with an
    irreversible effect on the world. Loop 2's rollback erases the transcript, not the timeline —
    so rebuilding a build that spoke would say it again, to the same peers, for one message. The
    `used_tools` guard (written for images, long before this) is what makes that impossible, and
    this pins it at the point of maximum consequence.
    """
    scripted = ScriptedMessages(platform, message(uuid=M0, body="what's the status?"))
    # The *post itself* is what makes a message arrive mid-build — the tightest possible race.
    speaking = _SpeakingArrivalTool(scripted, message(uuid=M1, body="landed mid-post"))
    client = BaseCradle(token=FAKE_TOKEN)
    harness = Harness(_ToolThenReplyProvider("messages"), tools=[speaking], home=tmp_path)
    agent = WakeAgent(harness, timeline=TIMELINE_UUID, client=client, onboard=False)

    posted = agent.wake()

    assert speaking.runs == 1  # said exactly once — the build was never rolled back and re-run
    assert len(posted) == 1
    assert _posts(platform) == ["Hello, John."]  # one body on the timeline, not two
    assert ClaimStore(tmp_path).claim(TIMELINE_UUID, M1, kind="messages") is True  # M1 → next wake


def test_a_probe_arriving_during_generation_is_acked_this_wake(platform, tmp_path):
    """A NOC probe landing mid-generation is acked THIS wake — its heartbeat stays sub-second.

    Loop 2 re-reads after generating; even when the only fresh item is a probe (not a peer, so
    no rebuild), it is absorbed and acked here rather than deferred to a later wake that may be
    serialized behind this one.
    """
    scripted = ScriptedMessages(platform, message(uuid=M0, body="a question"))  # human

    def on_chat(call_index):
        if call_index == 1:
            body = f"NOC message-seam probe — please disregard.\n{probe_marker()}"
            scripted.arrive(peer_ai_message(uuid=M1, body=body))

    agent, provider = build_wake(
        tmp_path, HookedProvider(on_chat=on_chat), probe_secret=PROBE_SECRET
    )

    agent.wake()

    assert len(provider.prompts) == 1  # a lone probe is not a peer → no rebuild
    bodies = [
        json.loads(call.request.content)["message"]["body"]
        for call in platform.post(f"/timelines/{TIMELINE_UUID}/messages").calls
    ]
    assert f"BCNOC1-ACK {PROBE_NONCE}" in bodies  # the probe was acked, not deferred
    assert MarkStore(tmp_path).get(TIMELINE_UUID) == M1  # and marked, so it is not re-acked


def test_the_settle_loop_is_bounded_by_max_builds(platform, tmp_path):
    """Loop 1 cannot hold the wake forever: a peer AI posting on every read caps at max_builds.

    With 3+ AI peers (or a peer whose own pacing is off) a newer AI message can land during
    every read window; an uncapped settle would hold the router lock indefinitely. The restart
    count is bounded by max_builds, after which the wake proceeds to generate.
    """
    counter = {"n": 0}
    scripted = ScriptedMessages(platform, peer_ai_message(uuid=M0, body="msg 0"))
    later = [M1, M2, M3, REPLY, "019e7756-aaaa-7aaa-8aaa-aaaaaaaaaaaa"]

    def recording_then_arrive(seconds):
        sleep(seconds)
        counter["n"] += 1
        # A newer peer AI lands during EVERY read window — the settle would never converge.
        if counter["n"] <= len(later):
            scripted.arrive(
                peer_ai_message(uuid=later[counter["n"] - 1], body=f"msg {counter['n']}")
            )

    sleep = RecordingSleep()
    pacer = ReadPacer(clock=lambda: PACE_CREATED, sleep=recording_then_arrive)
    agent, provider = build_wake(tmp_path, HookedProvider(on_chat=None), pacer=pacer, max_builds=3)

    posted = agent.wake()

    # The settle read-paces at most `max_builds` (3) times, then the cap stops it — a runaway
    # room can no longer hold the wake forever (without the cap this would loop indefinitely).
    assert len(sleep.calls) == 3
    assert len(provider.prompts) == 1  # it still runs one batched turn
    assert posted == []  # a silent brain


def test_bootstrap_recovers_an_orphaned_claim_instead_of_stranding_it(platform, tmp_path):
    """A first-wake message orphaned by a crashed prior wake is **answered**, not written off.

    This test used to assert the opposite, and its own fixture said so out loud — the message body
    was `"unanswerable — claim is orphaned"` and it pinned `posted == []`. The peer was never
    answered; the mark simply baselined past them so the agent would not re-bootstrap forever. The
    livelock guard was right; treating the message as unanswerable was the #285 defect in miniature.

    Now the orphan is recovered: the dead wake left no transcript turn for M0 (it died before the
    model ever saw it), so nothing ran and nothing posted, and re-driving is provably safe. The
    livelock guard still holds — the mark advances — but it advances because the message was
    *answered*, not because it was abandoned.
    """
    ClaimStore(tmp_path).claim(TIMELINE_UUID, M0, kind="messages")  # a dead wake's orphaned claim
    serve_messages(platform, page(message(uuid=M0, body="a peer's message the dead wake took")))
    agent, provider = build_wake(tmp_path)

    posted = agent.wake()

    assert len(posted) == 1  # the peer IS answered — the whole point of #285
    assert provider.prompts, "the orphan must be re-driven through the model, not written off"
    assert MarkStore(tmp_path).get(TIMELINE_UUID) == M0  # and the mark advances → no re-bootstrap
    assert ClaimStore(tmp_path).read(TIMELINE_UUID, M0, kind="messages").phase == "done"

    # A second wake is now a normal incremental one (mark present, claim settled) and does nothing.
    serve_messages(platform, page(message(uuid=M0, body="a peer's message the dead wake took")))
    second, second_provider = build_wake(tmp_path)
    assert second.wake() == []
    assert second_provider.prompts == []  # settled: never answered twice


def test_pace_max_builds_env_tunable(monkeypatch):
    """`HARNESS_PACE_MAX_BUILDS` overrides the default (3); blank → default; non-positive → 1."""
    monkeypatch.delenv("HARNESS_PACE_MAX_BUILDS", raising=False)
    assert _pace_max_builds_from_env() == 3  # the real default

    monkeypatch.setenv("HARNESS_PACE_MAX_BUILDS", "5")
    assert _pace_max_builds_from_env() == 5

    monkeypatch.setenv("HARNESS_PACE_MAX_BUILDS", "")  # blank → default
    assert _pace_max_builds_from_env() == 3

    monkeypatch.setenv("HARNESS_PACE_MAX_BUILDS", "0")  # floored to 1 (the loop always runs once)
    assert _pace_max_builds_from_env() == 1


def test_parse_created_at_handles_z_suffix_and_naive_stamps():
    """The ISO parse normalizes a `Z` suffix (3.10-safe) and assumes UTC for a naive stamp."""
    z = _parse_created_at("2026-06-04T00:00:00.000Z")
    assert z == PACE_CREATED and z.tzinfo is not None

    offset = _parse_created_at("2026-06-04T00:00:00+00:00")
    assert offset == PACE_CREATED

    naive = _parse_created_at("2026-06-04T00:00:00")  # no offset → assumed UTC, aware
    assert naive == PACE_CREATED and naive.tzinfo is timezone.utc


# --- the wake's log trail (issue #272) ----------------------------------------
#
# A deployed wake's only witness is its journal, so these pin the lines an operator (and Better
# Stack's Live Tail) actually reads: the bookends around every wake, the delivery-id correlation
# the router threads through, and the three failure classes that used to pass in silence — a
# refused post, a step-cap degradation, and a wake that never ran at all.


def _lines(caplog, level=None) -> list[str]:
    return [r.getMessage() for r in caplog.records if level is None or r.levelname == level]


def _line(caplog, prefix: str) -> str:
    return next(m for m in _lines(caplog) if m.startswith(prefix))


def test_a_wake_is_bookended_by_a_start_and_an_end_line(platform, tmp_path, caplog):
    """The two lines a wake always leaves: what it is about to run, and what came of it."""
    serve_messages(platform, page(message(uuid=M0, body="What's the status?")))
    agent, _ = build_wake(tmp_path)

    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        agent.wake()

    start = _line(caplog, "wake start")
    assert f"timeline={TIMELINE_UUID}" in start
    assert "provider=openai" in start and "model=gpt-4o" in start

    end = _line(caplog, "wake end")
    assert "outcome=ok" in end
    assert "turns=1" in end  # one model turn — the whole unseen batch, engaged once
    # Two steps: the model called the `messages` tool, then settled on its narration. That is what
    # speaking costs since #293 — a reply is a tool call, and a tool call is a step.
    assert "steps=2/24" in end
    # `posted=1` counts what the agent *chose* to send, read off the speech ledger the tool records
    # into. A silent wake reports `posted=0` — visibly, deliberately silent.
    assert "posted=1" in end
    assert re.search(r"duration=\d+\.\d\ds", end)


def test_a_named_trigger_rides_the_start_line(platform, tmp_path, caplog):
    serve_messages(platform, page(message(uuid=M0, body="hi")))
    agent, _ = build_wake(tmp_path)

    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        agent.wake(trigger=M0)

    assert f"trigger=message:{M0}" in _line(caplog, "wake start")


def test_the_bookends_carry_the_routers_delivery_id_when_it_exports_one(
    platform, tmp_path, caplog, monkeypatch
):
    """The correlation field (basecradle-router#170): both bookends echo the delivery that
    spawned the wake, so a router-side line and a harness-side line join in Live Tail."""
    monkeypatch.setenv("BASECRADLE_DELIVERY_ID", "01996f0e-3d2b-7a41-9c5f-2e6a7b8c9d0e")
    serve_messages(platform, page(message(uuid=M0, body="hi")))
    agent, _ = build_wake(tmp_path)

    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        agent.wake()

    for line in (_line(caplog, "wake start"), _line(caplog, "wake end")):
        assert "delivery=01996f0e-3d2b-7a41-9c5f-2e6a7b8c9d0e" in line


def test_without_the_delivery_var_the_field_is_simply_absent(
    platform, tmp_path, caplog, monkeypatch
):
    """Optional-when-absent: the harness and the router ship in either order, and a hand-run
    wake logs a clean line rather than an empty `delivery=`."""
    monkeypatch.delenv("BASECRADLE_DELIVERY_ID", raising=False)
    serve_messages(platform, page(message(uuid=M0, body="hi")))
    agent, _ = build_wake(tmp_path)

    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        agent.wake()

    assert "delivery=" not in _line(caplog, "wake start")


def test_a_quiet_wake_reports_ok_with_nothing_posted(platform, tmp_path, caplog):
    serve_messages(platform, page(message(uuid=M0, body="hi", mine=True)))  # only its own post
    agent, provider = build_wake(tmp_path)

    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        agent.wake()

    assert provider.prompts == []  # no model call…
    end = _line(caplog, "wake end")
    assert "outcome=ok" in end and "posted=0" in end
    assert "turns=0" in end and "steps=0/24" in end  # the model was never engaged


def test_a_breaker_declined_wake_says_so_in_its_end_line(platform, tmp_path, caplog):
    """A self-declining wake is not a healthy one — the end line must not read `ok`."""
    clock = FakeClock()
    MarkStore(tmp_path).set(TIMELINE_UUID, M0)
    serve_messages(platform, page(message(uuid=M1, body="one"), message(uuid=M0, body="old")))
    for _ in range(2):  # burn the cap so the next wake trips
        _wake_with_breaker(tmp_path, CountingProvider(), clock, max_wakes=2).wake()

    agent = _wake_with_breaker(tmp_path, CountingProvider(), clock, max_wakes=2)
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        agent.wake()

    end = _line(caplog, "wake end")
    assert "outcome=declined" in end
    assert "steps=0/24" in end  # no provider call was made


def test_a_successful_post_logs_the_message_it_created(platform, tmp_path, caplog):
    """The intent line that replaces httpx's transport chatter: which message, on which
    timeline. It is what says the agent *spoke*, not merely that an HTTP call went out."""
    serve_messages(platform, page(message(uuid=M0, body="hi")))
    agent, _ = build_wake(tmp_path)

    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        agent.wake()

    posted = _line(caplog, "posted")
    assert f"message={REPLY}" in posted
    assert f"timeline={TIMELINE_UUID}" in posted


def test_a_refused_post_is_logged_at_error(platform, tmp_path, caplog):
    """A locked timeline: the agent thought, spent tokens, and could not speak. It degrades
    (exit 0, a transcript note) — which is exactly why the *log* has to be loud."""
    serve_messages(platform, page(message(uuid=M0, body="hi")))
    platform.post(f"/timelines/{TIMELINE_UUID}/messages").mock(
        return_value=httpx.Response(403, json=_locked_problem())
    )
    agent, _ = build_wake(tmp_path)

    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        agent.wake()

    errors = _lines(caplog, "ERROR")
    assert any(m.startswith("post failed") and TIMELINE_UUID in m for m in errors)
    assert "outcome=ok" in _line(caplog, "wake end")  # the wake itself still completed
    assert "posted=0" in _line(caplog, "wake end")  # …having delivered nothing


def test_the_step_cap_degradation_is_logged_at_warning(platform, tmp_path, caplog):
    """The canned "I got stuck" note posts, the item is marked seen, the process exits 0 — so
    without this WARNING a degraded wake was indistinguishable from a healthy one."""
    serve_messages(platform, page(message(uuid=M0, body="do something complicated")))

    class LoopingThenBoom:
        provider, model = "openai", "gpt-4o"

        def chat(self, messages, tools=None):
            if tools is None:  # the out-of-budget reserve call fails too
                raise RuntimeError("reserve model call failed")
            return Message.assistant(tool_calls=[ToolCall(id="call_1", name="noop", arguments={})])

    client = BaseCradle(token=FAKE_TOKEN)
    harness = Harness(LoopingThenBoom(), tools=[_NoopTool()], home=tmp_path, max_steps=2)
    agent = WakeAgent(harness, timeline=TIMELINE_UUID, client=client, onboard=False)

    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        agent.wake()

    warnings = _lines(caplog, "WARNING")
    assert any(m.startswith("degraded") and "reserve summary" in m for m in warnings)
    assert "steps=2/2" in _line(caplog, "wake end")  # the budget was genuinely spent


def test_main_logs_a_hard_startup_failure_at_error(platform, wake_env, monkeypatch, caplog):
    """The wake that never ran at all — previously a bare, unleveled `print` no severity filter
    could find. It still prints; it now also passes through the logger as an ERROR."""
    monkeypatch.delenv("AI_MODEL", raising=False)  # a hard config failure at provider build

    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        assert main(["--timeline", TIMELINE_UUID]) == 1

    errors = _lines(caplog, "ERROR")
    assert any(m.startswith("wake failed") and "AI_MODEL" in m for m in errors)


def test_httpx_is_demoted_so_its_transport_chatter_leaves_the_journal(monkeypatch):
    """The noise this replaces: one `INFO HTTP Request: …` per platform read, model call, and
    blob fetch. The harness's own lines say the same thing with the context that line lacked."""
    from basecradle_harness._basecradle import _configure_logging

    httpx_logger = logging.getLogger("httpx")
    saved = httpx_logger.level
    monkeypatch.delenv("HARNESS_LOG_LEVEL", raising=False)
    try:
        httpx_logger.setLevel(logging.NOTSET)
        with _isolated_root_logging():
            _configure_logging()
            assert httpx_logger.level == logging.WARNING
    finally:
        httpx_logger.setLevel(saved)


def test_a_debug_run_keeps_the_transport_lines(monkeypatch):
    """An operator who turned the level down to DEBUG is asking for the wire — leave httpx be."""
    from basecradle_harness._basecradle import _configure_logging

    httpx_logger = logging.getLogger("httpx")
    saved = httpx_logger.level
    monkeypatch.setenv("HARNESS_LOG_LEVEL", "DEBUG")
    try:
        httpx_logger.setLevel(logging.NOTSET)
        with _isolated_root_logging():
            _configure_logging()
            assert httpx_logger.level == logging.NOTSET  # untouched
    finally:
        httpx_logger.setLevel(saved)


def test_a_multi_item_wake_reports_its_turn_count_alongside_the_step_total(
    platform, tmp_path, caplog
):
    """`max_steps` is a *per-turn* budget, and a wake takes one turn per activated task — so the
    end line carries the turn count. Without it, a legitimate multi-item wake's cumulative step
    total would read as a blown budget."""
    serve_messages(platform, page())  # no messages — the tasks are the work
    serve_tasks(
        platform,
        task_page(
            task(uuid=T0, instructions="draft the release notes"),
            task(uuid=T1, instructions="check the mail"),
        ),
    )
    agent, provider = build_wake(tmp_path)

    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        agent.wake()

    assert len(provider.prompts) == 2  # two tasks → two model turns
    end = _line(caplog, "wake end")
    assert "turns=2" in end
    assert "steps=4/24" in end  # two steps each (tool call + settle), summed across the two turns
    assert "posted=2" in end


def test_a_crashing_wake_still_reports_what_it_had_done(platform, tmp_path, caplog):
    """The end line rides a `finally` — a wake that dies mid-reconcile is the one whose outcome
    matters most, and it must not vanish from the journal."""
    serve_messages(platform, page(message(uuid=M0, body="hi")))

    class Exploding:
        provider, model = "openai", "gpt-4o"

        def chat(self, messages, tools=None):
            raise RuntimeError("the brain caught fire")

    client = BaseCradle(token=FAKE_TOKEN)
    harness = Harness(Exploding(), home=tmp_path)
    agent = WakeAgent(harness, timeline=TIMELINE_UUID, client=client, onboard=False)

    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        with pytest.raises(RuntimeError):  # the failure still propagates — nothing is swallowed
            agent.wake()

    end = _line(caplog, "wake end")
    assert "outcome=error" in end
    assert "posted=0" in end


def test_the_breaker_alert_is_a_log_line_not_a_post(platform, tmp_path, caplog):
    """The breaker speaks to the **operator**, in the journal — never to the peers (issue #293).

    Its trip alert used to be a message on the timeline, written in the agent's own voice ("I
    appear to be in a wake loop here…") — words the agent never wrote and never chose to send. The
    harness does not speak for the agent any more, so the alert is the WARNING it always also was.
    Nothing is lost: the breadcrumb was always the operator's, and the NOC alerts on this exact
    string (`Wake breaker TRIPPED`), never on the post.
    """
    clock = FakeClock()
    MarkStore(tmp_path).set(TIMELINE_UUID, M0)
    serve_messages(platform, page(message(uuid=M1, body="one"), message(uuid=M0, body="old")))
    for _ in range(2):  # burn the cap so the next wake trips
        _wake_with_breaker(tmp_path, CountingProvider(), clock, max_wakes=2).wake()
    said_before = _posts(platform)

    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        _wake_with_breaker(tmp_path, CountingProvider(), clock, max_wakes=2).wake()

    assert "Wake breaker TRIPPED" in _line(caplog, "Wake breaker TRIPPED")  # the operator is told
    assert _posts(platform) == said_before  # …and the tripped wake put nothing on the timeline
    end = _line(caplog, "wake end")
    assert "outcome=declined" in end and "posted=0" in end


def test_a_probe_ack_is_logged_as_an_ack_not_as_the_agent_speaking(platform, tmp_path, caplog):
    """The probe seam is trace-free in the *transcript*, never in the log — but a heartbeat ack
    must not read as the agent talking, so it carries its own kind."""
    body = f"NOC message-seam probe — please disregard.\n{probe_marker()}"
    serve_messages(platform, page(message(uuid=M0, body=body)))
    agent, provider = build_wake(tmp_path, probe_secret=PROBE_SECRET)

    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        agent.wake()

    assert provider.prompts == []  # token-free: the model was never engaged
    assert "kind=probe-ack" in _line(caplog, "posted")


# --- issue #285: a hard-failed wake no longer drops the peer's message -------
#
# The guarantee: **at-least-once for the read, at-most-once for every side effect, exactly-once
# for the reply.** A message is re-driven only when the dead wake is *provably* side-effect-free;
# where it is not, the drop is loud. These are the contract — one per outcome of the classifier,
# plus the two the mechanism could plausibly get wrong (a live concurrent wake; a legacy claim).


def crashed_wake_owning(home, *uuids, phase="in-flight", kind="messages"):
    """A claim left behind by a wake that died: in-flight, owned by a process that is gone.

    pid 1 is never this test process, and `_orphaned` reads a claim whose pid is not ours and
    whose owner it can still see as alive only against the age backstop — so the record is stamped
    ancient. Together: unambiguously orphaned, without spawning and killing a real process.
    """
    store = ClaimStore(home)
    for uuid in uuids:
        store.claim(TIMELINE_UUID, uuid, kind=kind)
        store._write(
            store._path(TIMELINE_UUID, kind, uuid),
            Claim(phase=phase, pid=1, wake="a-wake-that-died", at=0.0),
        )
    return store


def live_wake_owning(home, *uuids, kind="messages"):
    """The mirror of `crashed_wake_owning`: a claim a **concurrent, still-running** wake holds.

    The same pid-1 trick, stamped *now* rather than ancient — so the age backstop does not fire and
    `_orphaned` answers no. It cannot be faked with this process's own pid: a claim from a
    *different wake of the same process* is orphaned by definition (a process runs its wakes one at
    a time, so that wake is over), which is exactly the case `_orphaned` is built to catch.
    """
    store = ClaimStore(home)
    for uuid in uuids:
        store.claim(TIMELINE_UUID, uuid, kind=kind)
        store._write(
            store._path(TIMELINE_UUID, kind, uuid),
            Claim(phase="in-flight", pid=1, wake="a-wake-still-running", at=time.time()),
        )
    return store


def test_a_wake_that_died_after_a_tool_ran_is_resumed_never_abandoned(platform, tmp_path, caplog):
    """Outcome 3: tools fired, no narration. The turn is **finished**, not dropped (issue #297).

    This test used to pin the opposite, and its old name said so — *"abandons the message loudly"*.
    That was right on the reasoning and wrong on the options: re-driving would re-fire the dead
    turn's tools, so dropping the peer was the least-bad *remaining* choice. There is a third one
    now. The turn's tool results are **on disk** (the transcript persists as the turn runs), so it
    needs neither re-running nor abandoning — it needs **continuing**. The model is handed the
    partial transcript and finishes the turn it started.

    The two things that must both be true, and they are the whole guarantee: the model **is**
    engaged (the peer is answered), and the tool is **not** run again (nothing double-fires).
    """
    serve_messages(platform, page(message(uuid=M0, body="generate me an image")))
    tool = _NoopTool()

    class RunsAToolThenDies:
        provider, model = "openai", "gpt-4o"

        def __init__(self):
            self.calls = 0

        def chat(self, messages, tools=None):
            self.calls += 1
            if self.calls == 1:
                return Message.assistant(
                    tool_calls=[ToolCall(id="call_1", name="noop", arguments={})]
                )
            raise RuntimeError("killed after the tool had already run")

    client = BaseCradle(token=FAKE_TOKEN)
    harness = Harness(RunsAToolThenDies(), tools=[tool], home=tmp_path)
    agent = WakeAgent(harness, timeline=TIMELINE_UUID, client=client, onboard=False)
    with pytest.raises(RuntimeError):
        agent.wake()
    assert tool.runs == 1  # the dead wake really did run it

    # The next wake finds an orphaned claim whose transcript shows a tool call was issued.
    serve_messages(platform, page(message(uuid=M0, body="generate me an image")))
    resumed = _NoopTool()

    class FinishesTheTurn:
        """Picks the turn up where it was cut off: speaks, then settles."""

        provider, model = "openai", "gpt-4o"

        def __init__(self):
            self.calls = 0
            self.seen: list[list[Message]] = []

        def chat(self, messages, tools=None):
            self.calls += 1
            self.seen.append(list(messages))
            if self.calls == 1:
                return Message.assistant(
                    tool_calls=[
                        ToolCall(
                            id="call_2",
                            name="messages",
                            arguments={"action": "create", "body": "Here it is."},
                        )
                    ]
                )
            return Message.assistant(content="Finished the turn the last wake started.")

    second, brain = build_wake(tmp_path, FinishesTheTurn(), tools=[MessagesTool(), resumed])
    with caplog.at_level(logging.WARNING):
        posted = second.wake()

    # It is finished, not written off: the model is engaged and the peer gets their answer.
    assert len(posted) == 1
    assert _posts(platform) == ["Here it is."]
    # And the dead turn's tool is **not** run a second time — its result was already on disk, and
    # was replayed to the model rather than re-executed. This is the whole promise.
    assert resumed.runs == 0
    # The model was handed the interrupted turn itself: the peer's message and the dead wake's own
    # tool call and result are all in front of it, with no *new* user turn appended.
    replayed = brain.seen[0]
    assert any(m.role == "user" and "generate me an image" in (m.content or "") for m in replayed)
    assert any(m.tool_calls and m.tool_calls[0].name == "noop" for m in replayed)
    assert any(r.message.startswith("resuming") for r in caplog.records)
    claim = ClaimStore(tmp_path).read(TIMELINE_UUID, M0, kind="messages")
    assert claim.phase == "done"  # the turn finished, so the claim is settled — not abandoned
    assert MarkStore(tmp_path).get(TIMELINE_UUID) == M0


def test_a_completed_turn_is_committed_never_re_driven(platform, tmp_path, caplog):
    """Outcome 2 — **the turn-commit anchor** (issue #293). A turn that finished is simply done.

    The dead wake reached its terminal narration, so the model *finished*: everything it decided to
    do on the timeline, it did (through its tools), and everything it decided not to do, it decided.
    The recovering wake makes **no model call and no post** — it commits the claim and moves on.

    This replaces the body-matching reconciliation the old design needed ("is one of my own posts,
    newer than this message, carrying exactly this body?"). That question only existed because the
    harness held a generated reply it still had to deliver. It holds nothing now, so the question
    is gone — along with its residual false-match between two identical bodies.
    """
    crashed_wake_owning(tmp_path, M0)
    # The dead wake's transcript: it read M0, spoke through its tool, and settled on a narration —
    # i.e. it *finished*, and only the claim-commit never happened.
    session = Harness(CountingProvider(), home=tmp_path).session(f"timeline:{TIMELINE_UUID}")
    session.history.append(Message.user(content="[2026-06-04T00:00:00.000Z] john: what's up?"))
    session.history.append(
        Message.assistant(
            tool_calls=[
                ToolCall(
                    id="c1", name="messages", arguments={"action": "create", "body": "Hello, John."}
                )
            ]
        )
    )
    session.history.append(Message.tool(tool_call_id="c1", content="Posted to timeline."))
    session.history.append(Message.assistant(content="Answered him. Nothing further."))
    session._save()
    serve_messages(
        platform,
        page(
            message(uuid=REPLY, body="Hello, John.", mine=True),
            message(uuid=M0, body="what's up?"),
        ),
    )
    MarkStore(tmp_path).set(TIMELINE_UUID, M1)  # a mark older than M0, so M0 is re-read

    agent, provider = build_wake(tmp_path)
    with caplog.at_level(logging.INFO):
        posted = agent.wake()

    assert provider.prompts == []  # not re-run: the turn had already finished
    assert posted == []  # and above all: never said twice
    assert _posts(platform) == []  # nothing new reached the timeline
    recovered = next(r.message for r in caplog.records if r.message.startswith("recovered"))
    assert M0 in recovered and "turn-completed" in recovered  # the audit trail, by its real reason
    assert ClaimStore(tmp_path).read(TIMELINE_UUID, M0, kind="messages").phase == "done"


def test_a_silent_completed_turn_is_committed_too(platform, tmp_path):
    """The case the old design could not even represent: the dead wake finished, and said nothing.

    Under the auto-post there was no such state — a completed turn always produced a reply, so
    "finished" and "spoke" were the same fact, and recovery could look for the reply on the
    timeline. Silence-default separates them, and the commit record follows the *turn*, not the
    speech: a turn that ran to its narration is finished whether or not anyone heard it, and
    re-driving it would put a message in front of the model that it has already, deliberately,
    declined to answer.
    """
    crashed_wake_owning(tmp_path, M0)
    session = Harness(CountingProvider(), home=tmp_path).session(f"timeline:{TIMELINE_UUID}")
    session.history.append(Message.user(content="[2026-06-04T00:00:00.000Z] john: what's up?"))
    session.history.append(Message.assistant(content="Small talk, wrapping up. No reply needed."))
    session._save()
    serve_messages(platform, page(message(uuid=M0, body="what's up?")))
    MarkStore(tmp_path).set(TIMELINE_UUID, M1)

    agent, provider = build_wake(tmp_path)
    posted = agent.wake()

    assert provider.prompts == []  # no re-drive — its silence was a decision, not a gap
    assert posted == []
    assert ClaimStore(tmp_path).read(TIMELINE_UUID, M0, kind="messages").phase == "done"
    assert MarkStore(tmp_path).get(TIMELINE_UUID) == M0  # settled: the mark may pass it


def test_recovery_reads_the_transcript_never_the_timeline(platform, tmp_path):
    """A newer own post on the timeline is now *irrelevant* to recovery — the transcript decides.

    This was the sharpest edge of the old design. `_absorb` posts a NOC probe ack **before** the
    model call, so a wake that acked a probe, claimed a peer message and then died leaves an own
    post newer than a message it never answered; the cheap test ("any own post newer than X?")
    would have read that ack as the reply and dropped the peer. Body-equality closed that hole at
    the cost of comparing message bodies forever.

    The turn-commit anchor removes the question entirely: the transcript holds no turn for M0, so
    the model never saw it, so it is re-driven — and the ack sitting newer on the timeline can no
    longer confuse anything, because nothing reads the timeline to make this decision.
    """
    crashed_wake_owning(tmp_path, M0)
    serve_messages(
        platform,
        page(
            message(
                uuid=REPLY, body="ack: some-nonce", mine=True
            ),  # a newer own post — not a reply
            message(uuid=M0, body="a peer's real question"),
        ),
    )
    MarkStore(tmp_path).set(TIMELINE_UUID, M1)

    agent, provider = build_wake(tmp_path)
    posted = agent.wake()

    assert provider.prompts, "the peer's message must still be re-driven despite the newer own post"
    assert len(posted) == 1  # re-driven, and this time the agent spoke
    assert ClaimStore(tmp_path).read(TIMELINE_UUID, M0, kind="messages").phase == "done"


def test_a_live_concurrent_wake_still_owns_its_claim(platform, tmp_path):
    """Unchanged from before #285: a claim held by a *live* wake is skipped, never stolen.

    This is the race the `ClaimStore` has always closed, and recovery must not reopen it — a false
    "orphaned" would re-drive a message another wake is mid-way through answering, and post it
    twice. The claim is stamped with a **live pid that is not this process** (our parent) and a
    wake id that is not ours: a genuinely concurrent wake, by every test `_orphaned` applies.
    """
    store = ClaimStore(tmp_path)
    store.claim(TIMELINE_UUID, M0, kind="messages")
    store._write(
        store._path(TIMELINE_UUID, "messages", M0),
        Claim(phase="in-flight", pid=os.getppid(), wake="a-live-wake", at=time.time()),
    )
    serve_messages(platform, page(message(uuid=M0, body="mine, hands off")))
    MarkStore(tmp_path).set(TIMELINE_UUID, M1)

    agent, provider = build_wake(tmp_path)
    posted = agent.wake()

    assert provider.prompts == []  # skipped: another wake owns it
    assert posted == []
    assert ClaimStore(tmp_path).read(TIMELINE_UUID, M0, kind="messages").phase == "in-flight"
    # And the mark must NOT pass it. If that wake dies, the message has to stay findable — a mark
    # that sailed past an in-flight item would hide it forever, which is the bug #285 exists to fix.
    assert MarkStore(tmp_path).get(TIMELINE_UUID) == M1


def test_a_legacy_empty_claim_reads_as_done_and_is_never_re_driven(platform, tmp_path):
    """The fleet-upgrade case: @jt and @glm-5.2 have live claim dirs full of *empty* files.

    The pre-#285 code only ever created a claim for an item it was about to mark seen, so an empty
    claim means "the old code handled this" — final by construction. Reading it as `done`
    reproduces the old behavior exactly. Reading it as in-flight-and-orphaned would re-answer every
    message in every deployed agent's history on the first wake after the upgrade.
    """
    path = ClaimStore(tmp_path)._path(TIMELINE_UUID, "messages", M0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")  # exactly what the old code left behind

    assert ClaimStore(tmp_path).read(TIMELINE_UUID, M0, kind="messages").phase == "done"

    serve_messages(platform, page(message(uuid=M0, body="answered long ago")))
    MarkStore(tmp_path).set(TIMELINE_UUID, M1)
    agent, provider = build_wake(tmp_path)

    assert agent.wake() == []
    assert provider.prompts == []  # never re-answered


def test_the_bootstrap_baseline_never_sails_past_a_live_wakes_claim(platform, tmp_path):
    """The #285 bug's last hiding place: the *bootstrap* mark baseline.

    `_settle` holds the mark back from any message a live concurrent wake is still holding. But the
    first wake also **baselines** the mark to the newest message so it does not re-bootstrap forever
    — and an unconditional baseline steps straight over that held-back message, hiding it from every
    future wake if its owner then dies. Same defect, different door.

    Here a live wake owns M0 while this one bootstraps with M1 newer. The mark must **not** reach
    M1: doing so would put M0 permanently behind the cursor.
    """
    store = ClaimStore(tmp_path)
    store.claim(TIMELINE_UUID, M0, kind="messages")
    store._write(
        store._path(TIMELINE_UUID, "messages", M0),
        Claim(phase="in-flight", pid=os.getppid(), wake="a-live-wake", at=time.time()),
    )
    serve_messages(
        platform,
        page(message(uuid=M1, body="newer"), message(uuid=M0, body="held by a live wake")),
    )

    agent, _ = build_wake(tmp_path)  # no mark yet → the bootstrap path
    agent.wake()

    # The baseline declined: the mark never passed M0, so M0 survives its owner's death.
    assert MarkStore(tmp_path).get(TIMELINE_UUID) != M1
    assert ClaimStore(tmp_path).read(TIMELINE_UUID, M0, kind="messages").phase == "in-flight"

    # And it converges rather than livelocking: once the owner settles M0, the mark moves on.
    ClaimStore(tmp_path).commit(TIMELINE_UUID, M0, kind="messages")
    serve_messages(
        platform,
        page(message(uuid=M1, body="newer"), message(uuid=M0, body="held by a live wake")),
    )
    second, _ = build_wake(tmp_path)
    second.wake()
    assert MarkStore(tmp_path).get(TIMELINE_UUID) == M1


def test_the_bootstrap_recovers_a_dead_wakes_message_that_is_no_longer_the_newest(
    platform, tmp_path
):
    """The reachable, lossy variant: a dead wake's message is no longer the newest when we bootstrap.

    A wake claims M0 and dies. A new message M1 lands. The next wake has no mark, so it *bootstraps*
    — and `_bootstrap_split` answers "what is unseen?" by recency: never having spoken, it replies to
    the newest message only. M0 is seeded as *context* and the mark baselines past it. M0 is then
    behind the cursor, unanswered, and unreachable by every future wake: the exact #285 drop, walking
    in through the one door where the mark *jumps* instead of stepping.

    An unsettled claim is unfinished work, not history, so the reply set stretches back to cover it.
    """
    crashed_wake_owning(tmp_path, M0)  # a wake took M0 and died before answering it
    serve_messages(
        platform,
        page(message(uuid=M1, body="a newer question"), message(uuid=M0, body="the lost question")),
    )

    agent, provider = build_wake(tmp_path)  # no mark → bootstrap
    posted = agent.wake()

    # Both are answered — M0 is not written off as backlog just because M1 arrived after it.
    assert len(posted) == 1  # one batched reply covering both
    assert "the lost question" in provider.prompts[0]
    assert "a newer question" in provider.prompts[0]
    assert ClaimStore(tmp_path).read(TIMELINE_UUID, M0, kind="messages").phase == "done"
    assert MarkStore(tmp_path).get(TIMELINE_UUID) == M1


# --- the concurrency bugs the adversarial self-review found (issue #285) ------


def test_two_wakes_recovering_the_same_orphan_cannot_both_win_it(tmp_path):
    """`reclaim` must be atomic: an unlink-then-create take-over lets BOTH wakes win.

    The broken form is `unlink()` then exclusive-`claim()`, and it is safe only if *both* unlinks
    happen before *either* create. Interleaved, they do not: wake A unlinks, A creates (wins,
    writing a live claim) — then wake B unlinks, **destroying A's brand-new live claim**, and B
    creates and "wins" too. Both would drive the same message through the model and post it twice.

    The rename is the mutex: only one caller can move a given path away, so only one take-over can
    ever succeed. This drives the exact interleaving rather than hoping a scheduler produces it.
    """
    crashed_wake_owning(tmp_path, M0)
    a, b = ClaimStore(tmp_path), ClaimStore(tmp_path)

    dead = "a-wake-that-died"  # the owner both wakes judged orphaned
    won_a = a.reclaim(TIMELINE_UUID, M0, kind="messages", owner=dead)  # A takes it over...
    won_b = b.reclaim(TIMELINE_UUID, M0, kind="messages", owner=dead)  # ...then B tries the same

    assert won_a is True
    assert won_b is False, "B must not be able to steal A's live claim by deleting it"
    # And the surviving claim is A's — B did not overwrite it.
    assert a.read(TIMELINE_UUID, M0, kind="messages").wake == a.wake


def test_a_claim_is_never_observable_empty_and_so_never_misread_as_legacy_done(tmp_path):
    """A live in-flight claim must never be readable as the legacy `done` sentinel.

    `claim()` used to win the race with an exclusive create and write the record *after*, leaving a
    window where the path existed and the file was **zero bytes**. An empty claim is this store's
    legacy sentinel and reads as `done` — so a concurrent wake landing in that window would read a
    live, in-flight claim as *settled*, let the high-water mark pass it, and lose the peer's message
    forever if its owner then died. The file is now linked into place already carrying its record.
    """
    store = ClaimStore(tmp_path)
    assert store.claim(TIMELINE_UUID, M0, kind="messages") is True

    path = store._path(TIMELINE_UUID, "messages", M0)
    assert path.read_text().strip(), "the claim file must never be empty the instant it exists"
    claim = store.read(TIMELINE_UUID, M0, kind="messages")
    assert claim.phase == "in-flight" and claim.settled is False
    # No temp litter left behind by the link dance.
    assert [p.name for p in path.parent.iterdir() if p.name.startswith(".")] == []


def test_a_compaction_during_a_build_never_lets_its_tools_re_fire(platform, tmp_path):
    """A mid-build compaction invalidates the rollback point — so the build must be committed.

    `Session.send` may compact, and a compaction rewrites `history` **in place and shorter**. After
    it, `base_len` no longer points at the build's start: `history[base_len:]` comes back empty (so
    `used_tools` reads False even though a tool ran) and `del history[base_len:]` is a no-op (so the
    "rolled back" build is still there). Together they rebuild a tool-using turn and **re-fire its
    side effects** — a second image generated and posted for one request.

    Here the tool runs, the transcript is compacted, and a new peer message lands mid-generation.
    The tool must run exactly once.
    """
    tool_runs = []

    class CountingTool(Tool):
        name = "noop"
        description = "does nothing, but expensively"
        parameters = {"type": "object", "properties": {}}

        def run(self, **kwargs):
            tool_runs.append(1)
            return "done"

    class ToolThenReply:
        provider, model = "openai", "gpt-4o"

        def __init__(self):
            self.calls = 0
            self.last_tokens_in = None

        def chat(self, messages, tools=None):
            self.calls += 1
            if self.calls == 1:
                return Message.assistant(
                    tool_calls=[ToolCall(id="call_1", name="noop", arguments={})]
                )
            # Report huge usage so the post-turn compaction fires and rewrites history shorter.
            self.last_tokens_in = 900_000
            return Message.assistant(content="Hello, John.")

        def context_limit(self):
            return 128_000

    # A new peer message lands during generation → the staleness guard would want to rebuild.
    messages = ScriptedMessages(platform, message(uuid=M0, body="make me an image"))
    messages.arrive(message(uuid=M1, body="actually, wait"))

    client = BaseCradle(token=FAKE_TOKEN)
    harness = Harness(ToolThenReply(), tools=[CountingTool()], home=tmp_path)
    agent = WakeAgent(harness, timeline=TIMELINE_UUID, client=client, onboard=False)
    agent.wake()

    assert tool_runs == [1], "the tool must run exactly once — a rebuild would re-fire it"
