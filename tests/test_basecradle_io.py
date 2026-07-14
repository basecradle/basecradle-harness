"""The BaseCradle timeline I/O, against a respx-mocked platform.

No live platform call: respx stands in for the SDK's HTTP transport, returning
wire-shaped payloads (dashboard, timeline, message list, message create). The
fictional cast: Nova Digital (handle ``nova``, AI) is the agent; John Doe
(``john``, human) posts the messages it answers.
"""

import json
import os
import re
from pathlib import Path

import httpx
import pytest
import respx
from basecradle import BaseCradle

from basecradle_harness import (
    Harness,
    MemoryTool,
    Message,
    MessagesTool,
    OpenAIProvider,
    OpenRouterProvider,
    TimelineAgent,
    ToolCall,
    XaiSdkProvider,
    config_home,
    install,
)
from basecradle_harness._basecradle import (
    DEFAULT_CONTEXT_MESSAGES,
    DEFAULT_MAX_STEPS,
    DEFAULT_RESPONSE_RETRIES,
    _client_from_env,
    _compose_prompt,
    _config_from_env,
    _context_messages_from_env,
    _max_steps_from_env,
    _onboard_from_env,
    _orientation,
    _provider_from_config,
    _resolve_tools_and_provider,
    _response_retries_from_env,
    resolved_model_params,
)
from basecradle_harness._model_params import MODEL_PARAMS_NAME
from basecradle_harness._search_params import SEARCH_PARAMS_NAME
from basecradle_harness._version import __version__

BC_URL = "https://basecradle.com"
FAKE_TOKEN = "bc_uat_KqI8zFxkQ0OZ8vYwT7mWcVtR3nSdLpEa"
MINTED_TOKEN = "bc_uat_9mZ2pQ7rT4vW1xY6sLkN3bHcJ8dGfEa0"  # what login() returns

NOVA_UUID = "019e7750-66ee-79c8-ad8a-bbb6ea7c2bcc"  # the agent (me)
JOHN_UUID = "019e7750-66ee-7e50-9e54-3bf8c3d6a8f1"  # the human
TIMELINE_UUID = "019e7750-66ee-7f53-829f-13a8a710b6da"

# Well-formed UUIDv7 message ids, oldest → newest.
M0 = "019e7751-4a1b-7c2d-8e3f-1a2b3c4d5e6f"
M1 = "019e7752-5b2c-7d3e-9f40-2b3c4d5e6f70"
M2 = "019e7753-6c3d-7e4f-8051-3c4d5e6f7081"
M3 = "019e7754-7d4e-7e60-8172-4d5e6f708192"
REPLY = "019e7755-8e5f-7f70-9283-5e6f70819203"


# --- wire payload builders ---------------------------------------------------


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


def page(*messages, next_cursor=None):
    """One page of a cursor-paginated message list (last page unless given a cursor)."""
    return {"messages": list(messages), "next_cursor": next_cursor}


def dashboard():
    return {"identity": {"uuid": NOVA_UUID, "handle": "nova", "name": "Nova Digital", "kind": "ai"}}


def full_dashboard():
    """A Dashboard with the orientation sections a fresh peer wakes on."""
    return {
        "identity": {"uuid": NOVA_UUID, "handle": "nova", "name": "Nova Digital", "kind": "ai"},
        "environment": {
            "name": "BaseCradle",
            "summary": "a communications platform where humans and AI are peers",
            "you_are": "a first-class peer with your own timelines",
        },
        "documentation": {
            "user_guide": "https://basecradle.com/docs",
            "api": "https://basecradle.com/docs/api",
            "changelog": "https://basecradle.com/docs/changelog",
            "openapi": "https://basecradle.com/openapi.json",
            "reference": "https://basecradle.com/docs/reference",
        },
    }


def timeline():
    # The timeline-get envelope is two keys: the timeline subject and its items.
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


# --- a canned provider, so the agent has a brain without a model -------------


def _is_counter(m) -> bool:
    """A live step-counter note the engine injects before each provider call (issue #243).

    Matched by its `Step N of M` line — never the brief's `Current Time:` anchor or its
    `Step budget:` statement, whose wording is deliberately close but carries no `N of M`.
    """
    return m.role == "system" and bool(m.content) and bool(re.search(r"Step \d+ of \d+", m.content))


def _convo(messages: list) -> list:
    """The turns the model saw with the injected step-counter notes filtered out."""
    return [m for m in messages if not _is_counter(m)]


class CannedProvider:
    def __init__(self, text="Hello, John."):
        self.text = text
        self.prompts: list[str] = []
        self.last_messages: list = []  # the full context of the most recent call

    def chat(self, messages, tools=None):
        self.last_messages = list(messages)
        # Record the last real turn's text, skipping the engine's step-counter note (issue #243).
        self.prompts.append(_convo(messages)[-1].content)
        return Message.assistant(content=self.text)


class ToolCallingProvider:
    """Calls one tool on its first turn, then settles on plain text — an agent that *speaks*.

    The shape every real reply now takes (issue #293): the model acts through a tool call and then
    ends its turn with narration nobody but its operator reads.
    """

    def __init__(self, *, tool, arguments, text="Done."):
        self.tool = tool
        self.arguments = arguments
        self.text = text
        self.calls = 0

    def chat(self, messages, tools=None):
        self.calls += 1
        if self.calls == 1:
            return Message.assistant(
                tool_calls=[ToolCall(id="call-1", name=self.tool, arguments=self.arguments)]
            )
        return Message.assistant(content=self.text)


@pytest.fixture
def platform():
    with respx.mock(base_url=BC_URL, assert_all_called=False) as router:
        yield router


def wire(router, *, message_pages, dashboard_payload=None):
    """Register the four platform routes; `message_pages` drives the list endpoint.

    `dashboard_payload` overrides the (identity-only) default — pass
    `full_dashboard()` to exercise wake-on-dashboard onboarding.
    """
    router.get("/users/dashboard").mock(
        return_value=httpx.Response(200, json=dashboard_payload or dashboard())
    )
    router.get(f"/timelines/{TIMELINE_UUID}").mock(
        return_value=httpx.Response(200, json=timeline())
    )
    router.get("/messages").mock(side_effect=[httpx.Response(200, json=p) for p in message_pages])
    router.post(f"/timelines/{TIMELINE_UUID}/messages").mock(
        return_value=httpx.Response(
            201, json={"message": message(uuid=REPLY, body="reply", mine=True)}
        )
    )


def build_agent(provider=None, *, system_prompt=None, **kwargs):
    """Build the agent against an already-active respx. Returns (agent, provider).

    `system_prompt` seeds the Harness charter; extra kwargs (e.g. `context_messages`,
    `onboard`) pass straight through to TimelineAgent.
    """
    provider = provider or CannedProvider()
    client = BaseCradle(token=FAKE_TOKEN)
    harness = Harness(provider, system_prompt=system_prompt)
    agent = TimelineAgent(harness, timeline=TIMELINE_UUID, client=client, **kwargs)
    return agent, provider


# --- construction ------------------------------------------------------------


def test_construction_resolves_identity_and_high_water_mark(platform):
    wire(platform, message_pages=[page(message(uuid=M0, body="hi"))])
    agent, _ = build_agent()

    assert agent.me_uuid == NOVA_UUID
    assert agent._last_seen == M0


def test_construction_seeds_the_backlog_as_context(platform):
    """Prior timeline messages become conversation turns — the agent knows the history."""
    wire(
        platform,
        message_pages=[
            page(
                message(uuid=M2, body="anyone around?"),  # newest first on the wire
                message(uuid=M1, body="hi all", mine=True),
                message(uuid=M0, body="welcome"),
            )
        ],
    )
    agent, _ = build_agent()

    # Seeded oldest-first; others are user turns tagged with the speaker, the
    # agent's own posts are assistant turns.
    seeded = [(m.role, m.content) for m in agent.harness.history]
    assert seeded == [
        ("user", "[2026-06-04T00:00:00.000Z] john: welcome"),
        ("assistant", "hi all"),
        ("user", "[2026-06-04T00:00:00.000Z] john: anyone around?"),
    ]


def test_reply_sees_the_backlog_before_the_new_message(platform):
    """When the agent answers a new message, the prior context is in front of it."""
    wire(
        platform,
        message_pages=[
            page(message(uuid=M0, body="we were discussing Ruby")),  # backlog at startup
            page(
                message(uuid=M1, body="what did we decide?"),
                message(uuid=M0, body="we were discussing Ruby"),
            ),
        ],
    )
    provider = CannedProvider()
    agent, _ = build_agent(provider)

    agent.poll_once()

    context = [(m.role, m.content) for m in _convo(provider.last_messages)]
    assert context == [
        ("user", "[2026-06-04T00:00:00.000Z] john: we were discussing Ruby"),  # the backlog, seeded
        (
            "user",
            "[2026-06-04T00:00:00.000Z] john: what did we decide?",
        ),  # the new message it is answering
    ]


# --- bounding the seeded backlog (context cap) -------------------------------


def test_context_cap_seeds_only_the_most_recent_n(platform):
    """A finite cap seeds the most recent N messages, still oldest-first."""
    wire(
        platform,
        message_pages=[
            page(
                message(uuid=M3, body="newest"),  # newest first on the wire
                message(uuid=M2, body="third"),
                message(uuid=M1, body="second"),
                message(uuid=M0, body="oldest"),
            )
        ],
    )
    agent, _ = build_agent(context_messages=2)

    seeded = [(m.role, m.content) for m in agent.harness.history]
    assert seeded == [
        ("user", "[2026-06-04T00:00:00.000Z] john: third"),  # the most recent 2, oldest-first
        ("user", "[2026-06-04T00:00:00.000Z] john: newest"),
    ]
    assert agent._last_seen == M3  # high-water mark is still the true newest


def test_context_cap_none_seeds_the_whole_backlog(platform):
    """`None` opts back into seeding everything."""
    wire(
        platform,
        message_pages=[
            page(
                message(uuid=M2, body="third"),
                message(uuid=M1, body="second"),
                message(uuid=M0, body="oldest"),
            )
        ],
    )
    agent, _ = build_agent(context_messages=None)

    seeded = [m.content for m in agent.harness.history]
    assert seeded == [
        "[2026-06-04T00:00:00.000Z] john: oldest",
        "[2026-06-04T00:00:00.000Z] john: second",
        "[2026-06-04T00:00:00.000Z] john: third",
    ]


def test_context_cap_zero_seeds_nothing_but_keeps_high_water_mark(platform):
    """A cap of 0 seeds no context, yet still primes the mark to the newest."""
    wire(
        platform,
        message_pages=[
            page(message(uuid=M1, body="newest"), message(uuid=M0, body="oldest")),
            # second read (poll): nothing newer than M1
            page(message(uuid=M1, body="newest"), message(uuid=M0, body="oldest")),
        ],
    )
    provider = CannedProvider()
    agent, _ = build_agent(provider, context_messages=0)

    assert agent.harness.history == []  # no backlog seeded
    assert agent._last_seen == M1  # but the mark is the true newest

    # The cap governs context only — the agent still ignores the backlog as
    # "already seen" and replies to nothing here.
    assert agent.poll_once() == []
    assert provider.prompts == []


def test_context_cap_does_not_change_which_messages_get_engaged(platform):
    """Capping the seed never makes the agent engage on history it didn't seed."""
    wire(
        platform,
        message_pages=[
            page(  # startup: a backlog of three, capped to one
                message(uuid=M2, body="third"),
                message(uuid=M1, body="second"),
                message(uuid=M0, body="first"),
            ),
            page(  # poll: one genuinely new message on top of the same backlog
                message(uuid=M3, body="brand new"),
                message(uuid=M2, body="third"),
                message(uuid=M1, body="second"),
                message(uuid=M0, body="first"),
            ),
        ],
    )
    provider = CannedProvider()
    agent, _ = build_agent(provider, context_messages=1)

    agent.poll_once()

    assert provider.prompts == [
        "[2026-06-04T00:00:00.000Z] john: brand new"
    ]  # only the new message, not the backlog


def test_capped_seed_does_not_paginate_the_whole_timeline(platform):
    """An islice'd seed fetches only the pages it needs — not every page."""
    wire(
        platform,
        message_pages=[
            page(  # first page; a cursor means more pages exist behind it
                message(uuid=M3, body="newest"),
                message(uuid=M2, body="third"),
                next_cursor="019e7760-0000-7000-8000-000000000000",
            ),
            page(message(uuid=M1, body="second"), message(uuid=M0, body="oldest")),
        ],
    )
    build_agent(context_messages=2)

    # The cap is satisfied by the first page, so the second page is never fetched.
    assert platform.get("/messages").call_count == 1


def test_negative_context_messages_is_rejected():
    """The cap is validated before any network call — fail fast on nonsense."""
    with pytest.raises(ValueError):
        TimelineAgent(
            Harness(CannedProvider()),
            timeline=TIMELINE_UUID,
            client=BaseCradle(token=FAKE_TOKEN),
            context_messages=-1,
        )


# --- responding --------------------------------------------------------------


def test_a_turns_final_text_is_never_posted(platform):
    """The Unspoken Channel (issue #293): the model's final text reaches no timeline, ever.

    This is the inversion, pinned at its narrowest point. The agent is engaged on the message and
    answers with plain text — the exact shape that used to auto-post — and **nothing** is sent to
    the platform. That text is now unspoken: journaled for the operator, shown to the agent's own
    next turn, seen by no peer.
    """
    wire(
        platform,
        message_pages=[
            page(message(uuid=M0, body="old")),
            page(message(uuid=M1, body="What's the status?"), message(uuid=M0, body="old")),
        ],
    )
    provider = CannedProvider(text="All clear, John.")
    agent, _ = build_agent(provider)

    posted = agent.poll_once()

    assert posted == []  # the agent said nothing — it never called a tool
    assert provider.prompts == [
        "[2026-06-04T00:00:00.000Z] john: What's the status?"
    ]  # it *was* engaged on the message (tagged with the speaker) — it simply did not speak
    assert not platform.post(f"/timelines/{TIMELINE_UUID}/messages").called


def test_the_agent_speaks_by_calling_the_messages_tool(platform):
    """The other half of the inversion: a deliberate `messages` create is what reaches a peer.

    One speaking channel, and this is it. The body posted is the one the model passed to the tool
    — nothing the harness composed on its behalf — and it lands exactly once, even though the turn
    also ends with its usual narration (the double-post that started all this, issue #293).
    """
    wire(
        platform,
        message_pages=[
            page(message(uuid=M0, body="old")),
            page(message(uuid=M1, body="What's the status?"), message(uuid=M0, body="old")),
        ],
    )
    provider = ToolCallingProvider(
        tool="messages",
        arguments={"action": "create", "body": "All clear, John."},
        text="Answered him. Nothing further needed.",  # the narration — must NOT be posted
    )
    client = BaseCradle(token=FAKE_TOKEN)
    harness = Harness(provider, tools=[MessagesTool()])
    agent = TimelineAgent(harness, timeline=TIMELINE_UUID, client=client)

    posted = agent.poll_once()

    post_route = platform.post(f"/timelines/{TIMELINE_UUID}/messages")
    assert post_route.call_count == 1  # exactly once — the narration did not double it
    assert json.loads(post_route.calls.last.request.content) == {
        "message": {"body": "All clear, John."}
    }
    assert len(posted) == 1  # and the agent's own post is what the poll reports


def test_does_not_reply_to_its_own_messages(platform):
    wire(
        platform,
        message_pages=[
            page(message(uuid=M0, body="old")),
            page(message(uuid=M1, body="my own post", mine=True), message(uuid=M0, body="old")),
        ],
    )
    provider = CannedProvider()
    agent, _ = build_agent(provider)

    posted = agent.poll_once()

    assert posted == []
    assert provider.prompts == []
    assert not platform.post(f"/timelines/{TIMELINE_UUID}/messages").called


def test_no_new_messages_means_no_reply(platform):
    wire(
        platform,
        message_pages=[page(message(uuid=M0, body="old")), page(message(uuid=M0, body="old"))],
    )
    agent, provider = build_agent()

    assert agent.poll_once() == []
    assert provider.prompts == []


def test_multiple_new_messages_handled_oldest_first(platform):
    wire(
        platform,
        message_pages=[
            page(message(uuid=M0, body="old")),
            page(
                message(uuid=M2, body="second"),
                message(uuid=M1, body="first"),
                message(uuid=M0, body="old"),
            ),
        ],
    )
    provider = CannedProvider()
    agent, _ = build_agent(provider)

    agent.poll_once()

    assert provider.prompts == [
        "[2026-06-04T00:00:00.000Z] john: first",
        "[2026-06-04T00:00:00.000Z] john: second",
    ]  # chronological, speaker-tagged — one engaged turn each


# --- the poll loop -----------------------------------------------------------


def test_run_polls_the_requested_number_of_times(platform):
    wire(
        platform,
        message_pages=[page(message(uuid=M0, body="old"))] * 3,  # init + two polls, nothing new
    )
    agent, _ = build_agent()

    agent.run(interval=0, max_polls=2)

    assert platform.get("/messages").call_count == 3  # 1 priming + 2 polls


# --- env configuration -------------------------------------------------------


def test_from_env_wires_a_full_agent(platform, monkeypatch):
    monkeypatch.setenv("BASECRADLE_TOKEN", FAKE_TOKEN)
    monkeypatch.setenv("BASECRADLE_TIMELINE", TIMELINE_UUID)
    monkeypatch.setenv("AI_MODEL", "gpt-4o")
    monkeypatch.setenv("AI_API_KEY", "sk-test-key")
    wire(platform, message_pages=[page(message(uuid=M0, body="hi"))])

    agent = TimelineAgent.from_env()

    assert agent.timeline_uuid == TIMELINE_UUID
    assert agent.me_uuid == NOVA_UUID
    # The shipped memory tool is wired in by default.
    assert "memory" in agent.harness.tools
    assert isinstance(agent.harness.tools.get("memory"), MemoryTool)
    # And web_fetch — a plain tool (no platform binding), like memory.
    assert "web_fetch" in agent.harness.tools
    # The assets tool too, bound to this agent's client and current timeline.
    assert "assets" in agent.harness.tools
    assets = agent.harness.tools.get("assets")
    assert assets.bound is True
    assert assets.context.timeline == TIMELINE_UUID
    assert assets.context.client is agent.client
    # And the tasks tool, bound the same way.
    assert "tasks" in agent.harness.tools
    tasks = agent.harness.tools.get("tasks")
    assert tasks.bound is True
    assert tasks.context.timeline == TIMELINE_UUID
    assert tasks.context.client is agent.client
    # And the governance tools (timelines + trust), bound the same way.
    for name in ("timelines", "trust"):
        assert name in agent.harness.tools
        tool = agent.harness.tools.get(name)
        assert tool.bound is True
        assert tool.context.timeline == TIMELINE_UUID
        assert tool.context.client is agent.client
    # The powerful media tools (generate_image, listen) are opt-in everywhere (issue #168), so a
    # default-riding agent does NOT come up with them — they activate only via the tools/ overlay.
    assert "generate_image" not in agent.harness.tools
    assert "listen" not in agent.harness.tools


def test_resolve_tools_and_provider_flips_web_search_with_the_surface(monkeypatch):
    # The tool set + provider built-ins are plugin-resolved, so flipping the openai adapter's
    # surface changes the active set: web_search (a Responses-only built-in) is on under
    # `responses` and gone under `chat`. The function tools stay the same — behavior-preserving.
    # web_search and generate_image are powerful → opt-in (issue #168), so this opts them into the
    # persona's overlay to exercise the wiring.
    monkeypatch.setenv("AI_MODEL", "gpt-5.4-mini")
    monkeypatch.setenv("AI_API_KEY", "sk-test-key")
    install(os.environ["BASECRADLE_CONFIG_HOME"], opt_in=["web_search", "generate_image"])

    # Default config: openai provider, openai SDK, responses surface → web_search active.
    provider, resolved, _memory, _bridge = _resolve_tools_and_provider()
    assert isinstance(provider, OpenAIProvider)
    assert provider.surface == "responses"
    # web_search is enabled on the provider as a server-side built-in (driven by the plugin)…
    assert {"type": "web_search"} in provider._builtin_tools
    # …but it is never a function tool the engine runs.
    names = {t.name for t in resolved.tools}
    assert "web_search" not in names
    assert "generate_image" in names  # opted-in OpenAI power tool, key present → on
    # Memory is wired from the provider now, not a plugin, but still lands in the resolved set.
    assert "memory" in names
    # The manifest names the active tools — including the server-side built-in, which is not
    # a function tool — so the brief can list exactly what the model can call.
    manifest_names = {name for name, _ in resolved.manifest}
    assert {"web_search", "generate_image", "memory"} <= manifest_names
    provider.close()

    monkeypatch.setenv("AI_SDK_SURFACE", "chat")
    chat_provider, chat_resolved, _chat_memory, _chat_bridge = _resolve_tools_and_provider()
    assert isinstance(chat_provider, OpenAIProvider)
    assert chat_provider.surface == "chat"
    assert chat_provider._builtin_tools == []  # web_search self-excludes off Responses
    # Same function tools; the Responses-only built-in is gone.
    assert {t.name for t in chat_resolved.tools} == names
    assert "web_search" not in {name for name, _ in chat_resolved.manifest}
    chat_provider.close()


def test_resolve_surfaces_a_broken_shipped_default_not_silently(monkeypatch):
    # A shipped default that fails to import is surfaced in resolved.broken (→ the loud Turn-0
    # brief defect section) and dropped from the active set, never silently swallowed (issue #160).
    monkeypatch.setenv("AI_MODEL", "gpt-5.4-mini")
    monkeypatch.setenv("AI_API_KEY", "sk-test-key")
    home = config_home()  # the conftest-isolated temp config home
    install(home)  # stamps the current version, so the upgrade reconcile is a no-op here
    (home / "tools" / "web_fetch.py").write_text("import a_symbol_the_rebuild_removed_zzz\n")

    provider, resolved, _memory, _bridge = _resolve_tools_and_provider()
    try:
        assert any("web_fetch.py" in line for line in resolved.broken)  # surfaced as a defect
        assert "web_fetch" not in {t.name for t in resolved.tools}  # the broken tool is gone
    finally:
        provider.close()


def test_resolve_runs_the_upgrade_reconcile_before_loading_the_overlay(monkeypatch):
    # The @jt root-cause fix is wired in: resolution runs the upgrade reconcile first, so an
    # overlay left stale by a `pip install -U` (running version ≠ the stamp) is refreshed before
    # it is loaded. Faithful refresh-of-a-changed-default is covered in test_install.py via the
    # `defaults=` seam; here we pin that resolution actually *invokes* the reconcile, by the
    # observable side effect — the stamp is advanced back to the running version.
    from basecradle_harness import installed_version

    monkeypatch.setenv("AI_MODEL", "gpt-5.4-mini")
    monkeypatch.setenv("AI_API_KEY", "sk-test-key")
    home = config_home()
    install(home)
    (home / ".version").write_text("0.0.0\n")  # simulate a config home from an older harness
    assert installed_version(home) == "0.0.0"

    provider, _resolved, _memory, _bridge = _resolve_tools_and_provider()
    try:
        assert installed_version(home) == __version__  # the reconcile ran and re-stamped
    finally:
        provider.close()


def test_from_env_honors_the_context_messages_cap(platform, monkeypatch):
    monkeypatch.setenv("BASECRADLE_TOKEN", FAKE_TOKEN)
    monkeypatch.setenv("BASECRADLE_TIMELINE", TIMELINE_UUID)
    monkeypatch.setenv("AI_MODEL", "gpt-4o")
    monkeypatch.setenv("AI_API_KEY", "sk-test-key")
    monkeypatch.setenv("HARNESS_CONTEXT_MESSAGES", "1")
    wire(
        platform,
        message_pages=[page(message(uuid=M1, body="newest"), message(uuid=M0, body="oldest"))],
    )

    agent = TimelineAgent.from_env()

    seeded = [m.content for m in agent.harness.history]
    assert seeded == [
        "[2026-06-04T00:00:00.000Z] john: newest"
    ]  # the env cap of 1 took only the most recent
    assert agent._last_seen == M1


def test_context_messages_from_env_parses_int_all_and_default(monkeypatch):
    monkeypatch.delenv("HARNESS_CONTEXT_MESSAGES", raising=False)
    assert _context_messages_from_env() == DEFAULT_CONTEXT_MESSAGES  # unset → default

    monkeypatch.setenv("HARNESS_CONTEXT_MESSAGES", "all")
    assert _context_messages_from_env() is None  # sentinel → seed everything

    monkeypatch.setenv("HARNESS_CONTEXT_MESSAGES", "ALL")
    assert _context_messages_from_env() is None  # case-insensitive

    monkeypatch.setenv("HARNESS_CONTEXT_MESSAGES", "7")
    assert _context_messages_from_env() == 7


def test_max_steps_from_env_parses_default_override_and_rejects_non_positive(monkeypatch):
    monkeypatch.delenv("HARNESS_MAX_STEPS", raising=False)
    assert _max_steps_from_env() == DEFAULT_MAX_STEPS  # unset → the shipped budget (24)

    monkeypatch.setenv("HARNESS_MAX_STEPS", "  ")
    assert _max_steps_from_env() == DEFAULT_MAX_STEPS  # blank → default too

    monkeypatch.setenv("HARNESS_MAX_STEPS", "40")
    assert _max_steps_from_env() == 40  # the operator's per-persona override

    monkeypatch.setenv("HARNESS_MAX_STEPS", "0")
    with pytest.raises(ValueError, match="positive integer"):
        _max_steps_from_env()  # a budget of 0 could never make a call — fail loudly


def test_response_retries_from_env_parses_default_override_zero_and_rejects_negative(monkeypatch):
    monkeypatch.delenv("HARNESS_RESPONSE_RETRIES", raising=False)
    assert _response_retries_from_env() == DEFAULT_RESPONSE_RETRIES  # unset → the shipped 2

    monkeypatch.setenv("HARNESS_RESPONSE_RETRIES", "  ")
    assert _response_retries_from_env() == DEFAULT_RESPONSE_RETRIES  # blank → default too

    monkeypatch.setenv("HARNESS_RESPONSE_RETRIES", "5")
    assert _response_retries_from_env() == 5  # the operator's per-persona override

    monkeypatch.setenv("HARNESS_RESPONSE_RETRIES", "0")
    assert _response_retries_from_env() == 0  # zero is valid — disable the retry (a single attempt)

    monkeypatch.setenv("HARNESS_RESPONSE_RETRIES", "-1")
    with pytest.raises(ValueError, match="non-negative integer"):
        _response_retries_from_env()  # a negative would silently mean "no attempts" — fail loudly


# --- config + provider selection (AI_PROVIDER / AI_SDK / AI_SDK_SURFACE) ---


def _set_model_key(monkeypatch):
    monkeypatch.setenv("AI_MODEL", "gpt-5.4-mini")
    monkeypatch.setenv("AI_API_KEY", "sk-test-key")


def test_config_from_env_defaults_to_the_openai_sdk_responses_stack(monkeypatch):
    for var in ("AI_PROVIDER", "AI_SDK", "AI_SDK_SURFACE"):
        monkeypatch.delenv(var, raising=False)
    assert _config_from_env() == ("openai", "openai", "responses")


def test_config_from_env_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "XAI")
    monkeypatch.setenv("AI_SDK_SURFACE", "Chat")
    provider, _sdk, surface = _config_from_env()
    assert provider == "xai"
    assert surface == "chat"


def test_config_from_env_rejects_an_unknown_provider(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "telepathy")
    with pytest.raises(ValueError, match="AI_PROVIDER"):
        _config_from_env()


def test_config_from_env_omitted_surface_uses_the_adapter_default(monkeypatch):
    # Omitted → the active SDK adapter's DEFAULT_SURFACE (openai → responses), not a global one.
    for var in ("AI_SDK", "AI_SDK_SURFACE"):
        monkeypatch.delenv(var, raising=False)
    _provider, _sdk, surface = _config_from_env()
    assert surface == "responses"


def test_config_from_env_rejects_an_unknown_surface(monkeypatch):
    # A typo is caught by validating against the active adapter's SURFACES (the hard-fail path).
    monkeypatch.setenv("AI_SDK_SURFACE", "interpretive-dance")
    with pytest.raises(ValueError, match="AI_SDK_SURFACE"):
        _config_from_env()


def test_config_from_env_xai_sdk_single_native_surface(monkeypatch):
    # The native xai-sdk declares one surface ("native"); AI_SDK_SURFACE is left unset → it
    # resolves to that default, and any *other* value fails clearly (issue #165).
    monkeypatch.setenv("AI_SDK", "xai-sdk")
    monkeypatch.delenv("AI_SDK_SURFACE", raising=False)
    assert _config_from_env() == ("openai", "xai-sdk", "native")

    monkeypatch.setenv("AI_SDK_SURFACE", "responses")
    with pytest.raises(ValueError, match="AI_SDK_SURFACE"):
        _config_from_env()


def test_provider_from_config_xai_sdk_builds_the_native_adapter(monkeypatch):
    # AI_SDK=xai-sdk + AI_PROVIDER=xai → the native gRPC adapter (issue #165).
    monkeypatch.setenv("AI_MODEL", "grok-4.3")
    monkeypatch.setenv("AI_API_KEY", "xai-test-key")
    provider = _provider_from_config("xai", "xai-sdk", "native", builtins=["web_search"])
    assert isinstance(provider, XaiSdkProvider)
    assert provider._builtin_tools == ["web_search"]
    provider.close()


def test_provider_from_config_xai_sdk_requires_the_xai_provider(monkeypatch):
    # The native SDK only reaches xAI's endpoint — pairing it with another provider fails clearly.
    monkeypatch.setenv("AI_MODEL", "grok-4.3")
    monkeypatch.setenv("AI_API_KEY", "xai-test-key")
    with pytest.raises(ValueError, match="AI_PROVIDER=xai"):
        _provider_from_config("openai", "xai-sdk", "native")


def test_provider_from_config_openai_builds_the_sdk_adapter(monkeypatch):
    _set_model_key(monkeypatch)
    provider = _provider_from_config("openai", "openai", "responses")
    assert isinstance(provider, OpenAIProvider)
    assert provider.surface == "responses"
    provider.close()


def test_provider_from_config_openai_honors_the_chat_surface(monkeypatch):
    _set_model_key(monkeypatch)
    provider = _provider_from_config("openai", "openai", "chat")
    assert isinstance(provider, OpenAIProvider)
    assert provider.surface == "chat"
    provider.close()


def test_provider_from_config_rejects_an_unimplemented_sdk(monkeypatch):
    # Milestone 1 ships only the openai adapter; another SDK name is a clear "no adapter yet".
    _set_model_key(monkeypatch)
    with pytest.raises(ValueError, match="AI_SDK"):
        _provider_from_config("openai", "anthropic", "responses")


def test_provider_from_config_requires_a_model(monkeypatch):
    monkeypatch.delenv("AI_MODEL", raising=False)
    monkeypatch.setenv("AI_API_KEY", "sk-test-key")
    with pytest.raises(ValueError, match="AI_MODEL"):
        _provider_from_config("openai", "openai", "responses")


def test_provider_from_config_passes_base_url_through(monkeypatch):
    _set_model_key(monkeypatch)
    monkeypatch.setenv("AI_BASE_URL", "https://openai-proxy.internal/v1")
    provider = _provider_from_config("openai", "openai", "responses")
    assert provider.base_url == "https://openai-proxy.internal/v1"
    provider.close()


def test_provider_from_config_xai_builds_the_openai_sdk_at_api_x_ai(monkeypatch):
    # xAI via the real openai SDK (issue #163): OpenAIProvider, base_url defaulted to api.x.ai.
    monkeypatch.setenv("AI_MODEL", "grok-4.3")
    monkeypatch.setenv("AI_API_KEY", "xai-test-key")
    monkeypatch.delenv("AI_BASE_URL", raising=False)

    provider = _provider_from_config(
        "xai", "openai", "responses", builtins=["web_search", "x_search"]
    )

    assert isinstance(provider, OpenAIProvider)
    assert provider.base_url == "https://api.x.ai/v1"
    # xAI's Live Search rides search_parameters (extra_body), not OpenAI tools entries.
    assert provider._builtin_tools == []
    assert provider._extra_body == {
        "search_parameters": {
            "mode": "on",
            "sources": ["web", "x"],
            "return_citations": True,
        }
    }
    provider.close()


def test_provider_from_config_xai_honors_the_chat_surface(monkeypatch):
    # The xAI matrix: AI_SDK=openai + chat surface is valid (xAI's compat endpoint speaks both),
    # and Live Search still rides search_parameters (it works on chat too).
    monkeypatch.setenv("AI_MODEL", "grok-4.3")
    monkeypatch.setenv("AI_API_KEY", "xai-test-key")
    monkeypatch.delenv("AI_BASE_URL", raising=False)

    provider = _provider_from_config("xai", "openai", "chat", builtins=["web_search"])

    assert isinstance(provider, OpenAIProvider)
    assert provider.surface == "chat"
    assert provider.base_url == "https://api.x.ai/v1"
    assert provider._extra_body == {
        "search_parameters": {"mode": "on", "sources": ["web"], "return_citations": True}
    }
    provider.close()


def test_provider_from_config_xai_honors_an_explicit_base_url(monkeypatch):
    monkeypatch.setenv("AI_MODEL", "grok-4.3")
    monkeypatch.setenv("AI_API_KEY", "xai-test-key")
    monkeypatch.setenv("AI_BASE_URL", "https://xai-proxy.internal/v1")

    provider = _provider_from_config("xai", "openai", "responses")

    assert isinstance(provider, OpenAIProvider)
    assert provider.base_url == "https://xai-proxy.internal/v1"
    # No search built-ins active → no search_parameters sent.
    assert provider._extra_body is None
    provider.close()


# --- OpenRouter: the provider × SDK × surface matrix (issue #234) -------------


def _set_openrouter_model_key(monkeypatch):
    monkeypatch.setenv("AI_MODEL", "z-ai/glm-5.2")
    monkeypatch.setenv("AI_API_KEY", "sk-or-test-key")
    monkeypatch.delenv("AI_BASE_URL", raising=False)


def test_provider_from_config_openrouter_native_builds_the_adapter(monkeypatch):
    # AI_SDK=openrouter + AI_PROVIDER=openrouter → the native OpenRouter adapter, chat surface.
    _set_openrouter_model_key(monkeypatch)
    provider = _provider_from_config("openrouter", "openrouter", "chat")
    assert isinstance(provider, OpenRouterProvider)
    assert provider.model == "z-ai/glm-5.2"
    assert provider.base_url == "https://openrouter.ai/api/v1"
    provider.close()


def test_provider_from_config_openrouter_sdk_requires_the_openrouter_provider(monkeypatch):
    # The native SDK only reaches OpenRouter's endpoint — pairing it with another provider fails.
    _set_openrouter_model_key(monkeypatch)
    with pytest.raises(ValueError, match="AI_PROVIDER=openrouter"):
        _provider_from_config("openai", "openrouter", "chat")


def test_provider_from_config_openrouter_native_honors_an_explicit_base_url(monkeypatch):
    _set_openrouter_model_key(monkeypatch)
    monkeypatch.setenv("AI_BASE_URL", "https://or-proxy.internal/api/v1")
    provider = _provider_from_config("openrouter", "openrouter", "chat")
    assert isinstance(provider, OpenRouterProvider)
    assert provider.base_url == "https://or-proxy.internal/api/v1"
    provider.close()


def test_provider_from_config_openrouter_wires_web_search_and_search_params(monkeypatch):
    # The builder threads the opted-in `web_search` built-in + the operator's search_params.json
    # into the native adapter as the OpenRouter server tool (issue #237).
    _set_openrouter_model_key(monkeypatch)
    (config_home() / SEARCH_PARAMS_NAME).write_text(
        json.dumps({"engine": "exa", "max_results": 8}), encoding="utf-8"
    )
    provider = _provider_from_config("openrouter", "openrouter", "chat", builtins=["web_search"])
    assert provider._server_tools == [
        {"type": "openrouter:web_search", "parameters": {"engine": "exa", "max_results": 8}}
    ]
    provider.close()


def test_provider_from_config_openrouter_bare_web_search_without_params(monkeypatch):
    # No search_params.json → the bare server-tool object (OpenRouter's defaults ride).
    _set_openrouter_model_key(monkeypatch)
    provider = _provider_from_config("openrouter", "openrouter", "chat", builtins=["web_search"])
    assert provider._server_tools == [{"type": "openrouter:web_search"}]
    provider.close()


def test_provider_from_config_openrouter_ignores_search_params_when_web_search_off(monkeypatch):
    # search_params.json is read ONLY when web search is active: a malformed file must not fail the
    # wake of a default-riding agent that never opted the tool in (unlike model_params.json, which
    # is always relevant to the model call). This guards the coupling the self-review flagged.
    _set_openrouter_model_key(monkeypatch)
    (config_home() / SEARCH_PARAMS_NAME).write_text("[not, an, object]", encoding="utf-8")
    provider = _provider_from_config("openrouter", "openrouter", "chat")  # no web_search builtin
    assert provider._server_tools == []  # no server tool, and the malformed file was never read
    provider.close()


def test_provider_from_config_openrouter_via_openai_sdk_chat(monkeypatch):
    # AI_PROVIDER=openrouter + AI_SDK=openai (chat) → OpenAIProvider at openrouter.ai (issue #234).
    _set_openrouter_model_key(monkeypatch)
    provider = _provider_from_config("openrouter", "openai", "chat")
    assert isinstance(provider, OpenAIProvider)
    assert provider.surface == "chat"
    assert provider.base_url == "https://openrouter.ai/api/v1"
    # No server-side built-ins are wired on the OpenRouter-via-openai cell.
    assert provider._builtin_tools == []
    provider.close()


def test_provider_from_config_openrouter_via_openai_sdk_rejects_responses(monkeypatch):
    # OpenRouter's Responses API is beta upstream — the openai-SDK cell is chat-only, and the
    # openai SDK's own default surface is `responses`, so the error must name the fix.
    _set_openrouter_model_key(monkeypatch)
    with pytest.raises(ValueError, match="AI_SDK_SURFACE=chat"):
        _provider_from_config("openrouter", "openai", "responses")


def test_config_from_env_openrouter_sdk_single_chat_surface(monkeypatch):
    # AI_SDK=openrouter declares one surface ("chat"); AI_SDK_SURFACE is left unset → it resolves
    # to that default, and any *other* value (e.g. responses) fails clearly.
    monkeypatch.setenv("AI_PROVIDER", "openrouter")
    monkeypatch.setenv("AI_SDK", "openrouter")
    monkeypatch.delenv("AI_SDK_SURFACE", raising=False)
    assert _config_from_env() == ("openrouter", "openrouter", "chat")

    monkeypatch.setenv("AI_SDK_SURFACE", "responses")
    with pytest.raises(ValueError, match="AI_SDK_SURFACE"):
        _config_from_env()


# --- model_params.json: the operator-owned parameter passthrough (issue #234) ---


def _write_model_params(obj):
    """Write ``obj`` as ``model_params.json`` into the (isolated) config home."""
    path = config_home() / MODEL_PARAMS_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def test_model_params_reach_the_openai_responses_build(monkeypatch):
    _set_model_key(monkeypatch)
    _write_model_params({"temperature": 0.7, "max_tokens": 4096})
    provider = _provider_from_config("openai", "openai", "responses")
    assert provider._default_params == {"temperature": 0.7, "max_tokens": 4096}
    provider.close()


def test_model_params_reach_the_openai_chat_build(monkeypatch):
    _set_model_key(monkeypatch)
    _write_model_params({"top_p": 0.9})
    provider = _provider_from_config("openai", "openai", "chat")
    assert provider._default_params == {"top_p": 0.9}
    provider.close()


def test_model_params_reach_the_xai_sdk_build(monkeypatch):
    monkeypatch.setenv("AI_MODEL", "grok-4.3")
    monkeypatch.setenv("AI_API_KEY", "xai-test-key")
    _write_model_params({"reasoning": {"effort": "high"}})
    provider = _provider_from_config("xai", "xai-sdk", "native")
    assert provider._default_params == {"reasoning": {"effort": "high"}}
    provider.close()


def test_model_params_reach_the_openrouter_build(monkeypatch):
    _set_openrouter_model_key(monkeypatch)
    _write_model_params({"reasoning_effort": "high", "temperature": 0.2})
    provider = _provider_from_config("openrouter", "openrouter", "chat")
    assert provider._default_params == {"reasoning_effort": "high", "temperature": 0.2}
    provider.close()


def test_model_params_reserved_keys_are_stripped_with_warnings(monkeypatch, caplog):
    # Harness-owned keys never override wiring — they are popped with a WARNING (D3).
    _set_model_key(monkeypatch)
    _write_model_params(
        {"model": "hacker/override", "messages": ["nope"], "tools": ["nope"], "temperature": 0.3}
    )
    with caplog.at_level("WARNING"):
        provider = _provider_from_config("openai", "openai", "responses")
    # Only the genuine tuning key survives.
    assert provider._default_params == {"temperature": 0.3}
    assert provider.model == "gpt-5.4-mini"  # AI_MODEL wins, not the params 'model'
    text = caplog.text
    assert "model identity is AI_MODEL" in text
    assert "'messages'" in text
    assert "'tools'" in text
    provider.close()


def test_model_params_stream_is_stripped_on_the_openai_build(monkeypatch, caplog):
    # All adapters are non-streaming by contract; a params `stream` would crash the turn
    # (a streaming iterator has no `.model_dump()`), so it is stripped everywhere — not just on
    # the openrouter branch. Regression guard for the owned-set omission.
    _set_model_key(monkeypatch)
    _write_model_params({"stream": True, "temperature": 0.3})
    with caplog.at_level("WARNING"):
        provider = _provider_from_config("openai", "openai", "responses")
    assert provider._default_params == {"temperature": 0.3}
    assert "'stream'" in caplog.text
    provider.close()


def test_model_params_stream_is_stripped_on_the_xai_sdk_build(monkeypatch, caplog):
    monkeypatch.setenv("AI_MODEL", "grok-4.3")
    monkeypatch.setenv("AI_API_KEY", "xai-test-key")
    _write_model_params({"stream": True, "temperature": 0.3})
    with caplog.at_level("WARNING"):
        provider = _provider_from_config("xai", "xai-sdk", "native")
    assert provider._default_params == {"temperature": 0.3}
    assert "'stream'" in caplog.text
    provider.close()


def test_model_params_timeout_is_stripped_on_the_openrouter_build(monkeypatch, caplog):
    # `timeout` is a harness-owned constructor arg — stripped like on the xai-sdk branch, so a
    # non-numeric value can never reach `int(timeout * 1000)`. Regression guard for the owned-set.
    _set_openrouter_model_key(monkeypatch)
    _write_model_params({"timeout": "30s", "temperature": 0.2})
    with caplog.at_level("WARNING"):
        provider = _provider_from_config("openrouter", "openrouter", "chat")
    assert provider._default_params == {"temperature": 0.2}
    assert "'timeout'" in caplog.text
    provider.close()


def test_model_params_http_headers_is_stripped_on_the_openrouter_build(monkeypatch, caplog):
    # `http_headers` is harness-owned: the adapter passes it to `chat.send` itself, carrying the
    # routing-metadata header (issue #280). Un-owned, an operator key of the same name would arrive
    # as a *second* value for that keyword — `TypeError: got multiple values for keyword argument` —
    # which is not the "unexpected keyword" shape the error mapper reframes, so it would crash the
    # wake raw rather than producing an actionable message.
    _set_openrouter_model_key(monkeypatch)
    _write_model_params({"http_headers": {"X-Mine": "1"}, "temperature": 0.2})
    with caplog.at_level("WARNING"):
        provider = _provider_from_config("openrouter", "openrouter", "chat")
    assert provider._default_params == {"temperature": 0.2}
    assert "'http_headers'" in caplog.text
    provider.close()


def test_model_params_extra_headers_does_not_collide_with_the_routing_header(monkeypatch, caplog):
    # The crash this forecloses: `extra_headers` is harness wiring on the openai-SDK-at-OpenRouter
    # cell (the routing-metadata header, issue #280). Left splatted in with the operator's tuning it
    # would arrive as a *second* value for the same keyword — `TypeError: got multiple values for
    # keyword argument 'extra_headers'` — a raw crash at provider construction, not the
    # warned-and-dropped collision policy. It is lifted out and *merged* instead of confiscated,
    # because an operator's own headers still work on this adapter's other two endpoints.
    _set_openrouter_model_key(monkeypatch)
    _write_model_params({"extra_headers": {"X-Mine": "1"}, "temperature": 0.2})

    provider = _provider_from_config("openrouter", "openai", "chat")

    assert provider._default_params == {"temperature": 0.2}
    # Both headers ride: the operator's is kept, the harness's routing header is added.
    assert provider._client.default_headers["X-Mine"] == "1"
    assert provider._client.default_headers["X-OpenRouter-Metadata"] == "enabled"
    provider.close()


def test_model_params_cannot_unset_the_routing_header(monkeypatch, caplog):
    # An operator who overrode it would not break anything *visible* — they would silently blind the
    # fleet's routing review, which is the exact failure class issue #280 is about. Harness wins,
    # loudly.
    _set_openrouter_model_key(monkeypatch)
    _write_model_params({"extra_headers": {"X-OpenRouter-Metadata": "disabled"}})

    with caplog.at_level("WARNING"):
        provider = _provider_from_config("openrouter", "openai", "chat")

    assert provider._client.default_headers["X-OpenRouter-Metadata"] == "enabled"
    assert "'X-OpenRouter-Metadata'" in caplog.text
    provider.close()


def test_malformed_model_params_does_not_mask_a_config_mismatch(monkeypatch):
    # A config-shape error must win over a malformed model_params.json: the wake should point at
    # the real problem (the sdk/provider mismatch), not at a tuning file it would never use.
    monkeypatch.setenv("AI_MODEL", "z-ai/glm-5.2")
    monkeypatch.setenv("AI_API_KEY", "sk-or-test-key")
    (config_home() / MODEL_PARAMS_NAME).write_text("{not json", encoding="utf-8")
    # xai-sdk + openrouter is a config mismatch — that error must surface, not the JSON error.
    with pytest.raises(ValueError, match="AI_PROVIDER=xai"):
        _provider_from_config("openrouter", "xai-sdk", "native")


def test_model_params_extra_body_reaches_the_openai_build(monkeypatch):
    # On the openai SDK, extra_body is a real passthrough — an operator's extra_body arrives intact.
    _set_model_key(monkeypatch)
    _write_model_params({"extra_body": {"custom_field": 1}, "temperature": 0.5})
    provider = _provider_from_config("openai", "openai", "responses")
    assert provider._extra_body == {"custom_field": 1}
    assert provider._default_params == {"temperature": 0.5}
    provider.close()


def test_model_params_extra_body_merges_with_xai_search_parameters(monkeypatch, caplog):
    # xai + a search built-in composes search_parameters; an operator extra_body merges under it,
    # the harness value winning any overlapping key, with a WARNING on the overlap (D4).
    monkeypatch.setenv("AI_MODEL", "grok-4.3")
    monkeypatch.setenv("AI_API_KEY", "xai-test-key")
    monkeypatch.delenv("AI_BASE_URL", raising=False)
    _write_model_params(
        {"extra_body": {"search_parameters": {"mode": "off"}, "other": 2}, "temperature": 0.1}
    )
    with caplog.at_level("WARNING"):
        provider = _provider_from_config("xai", "openai", "chat", builtins=["web_search"])
    # The harness's search_parameters wins the overlap; the operator's non-overlapping key stays.
    assert provider._extra_body == {
        "search_parameters": {"mode": "on", "sources": ["web"], "return_citations": True},
        "other": 2,
    }
    assert provider._default_params == {"temperature": 0.1}
    assert "search_parameters" in caplog.text
    provider.close()


def test_model_params_extra_body_dropped_on_openrouter_with_warning(monkeypatch, caplog):
    # The openrouter SDK has no extra_body concept — an operator's extra_body is dropped + warned.
    _set_openrouter_model_key(monkeypatch)
    _write_model_params({"extra_body": {"x": 1}, "temperature": 0.4})
    with caplog.at_level("WARNING"):
        provider = _provider_from_config("openrouter", "openrouter", "chat")
    assert provider._default_params == {"temperature": 0.4}
    assert "extra_body" in caplog.text
    provider.close()


def test_model_params_malformed_file_fails_the_build(monkeypatch):
    # A present-but-malformed model_params.json is a hard error at build — the wake fails loudly.
    _set_model_key(monkeypatch)
    (config_home() / MODEL_PARAMS_NAME).write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match=MODEL_PARAMS_NAME):
        _provider_from_config("openai", "openai", "responses")


# --- resolved_model_params: the read-only introspection twin of the collision policy (issue #236) ---


def test_resolved_model_params_absent_file_is_empty_with_no_collisions():
    # No file → the feature is off: an empty object and nothing stripped, for every SDK.
    assert resolved_model_params("openai") == ({}, [])
    assert resolved_model_params("xai-sdk") == ({}, [])
    assert resolved_model_params("openrouter") == ({}, [])


def test_resolved_model_params_reports_loaded_object_verbatim_and_openai_collisions():
    # The loaded object is returned exactly as written (harness-owned keys included — reporting is
    # honest); `stripped` names the owned collisions the openai build would drop. `reasoning`/
    # `temperature` are genuine tuning and never collide.
    _write_model_params({"reasoning": {"effort": "high"}, "temperature": 0.2, "model": "sneaky"})
    loaded, stripped = resolved_model_params("openai")
    assert loaded == {"reasoning": {"effort": "high"}, "temperature": 0.2, "model": "sneaky"}
    assert stripped == ["model"]


def test_resolved_model_params_keys_off_the_sdk_not_the_provider():
    # The openai SDK serves openai/xai/openrouter alike, so its owned set applies whenever
    # AI_SDK=openai — `web_search_params` is owned only on the *native* openrouter SDK, so under
    # the openai SDK it is treated as ordinary tuning (no collision).
    _write_model_params({"web_search_params": {"q": "x"}, "temperature": 0.1})
    _loaded, stripped = resolved_model_params("openai")
    assert stripped == []
    # On the native openrouter SDK the same key is harness-owned and reported stripped.
    _loaded, stripped = resolved_model_params("openrouter")
    assert stripped == ["web_search_params"]


def test_resolved_model_params_extra_body_stripped_only_where_the_sdk_drops_it():
    # extra_body is a real passthrough on the openai SDK (not stripped), but the native xai-sdk and
    # openrouter builds warn-and-drop it — so introspection counts it stripped exactly there.
    _write_model_params({"extra_body": {"x": 1}, "temperature": 0.5})
    assert resolved_model_params("openai") == (
        {"extra_body": {"x": 1}, "temperature": 0.5},
        [],
    )
    assert resolved_model_params("xai-sdk")[1] == ["extra_body"]
    assert resolved_model_params("openrouter")[1] == ["extra_body"]


def test_resolved_model_params_malformed_file_raises_naming_the_file():
    # Same failure a wake hits, surfaced at verify time: a malformed file is a loud ValueError.
    (config_home() / MODEL_PARAMS_NAME).write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match=MODEL_PARAMS_NAME):
        resolved_model_params("openai")


def test_model_params_land_in_the_request_body_end_to_end(monkeypatch):
    # The whole chain: model_params.json → _provider_from_config → the SDK's actual request body.
    _set_model_key(monkeypatch)
    monkeypatch.setenv("AI_BASE_URL", "https://openai.test/v1")
    _write_model_params({"temperature": 0.7, "max_tokens": 4096})
    provider = _provider_from_config("openai", "openai", "chat")
    with respx.mock(assert_all_called=True) as mock:
        route = mock.post("https://openai.test/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "chatcmpl-1",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "gpt-5.4-mini",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "ok"},
                        }
                    ],
                },
            )
        )
        provider.chat([Message.user("Hi")])
    body = json.loads(route.calls.last.request.content)
    assert body["temperature"] == 0.7
    assert body["max_tokens"] == 4096
    provider.close()


def test_from_env_wires_the_xai_profile_with_live_search_builtins(platform, monkeypatch):
    """End to end: AI_PROVIDER=xai gives Eddie the openai-SDK brain at xAI + Live Search.

    Eddie's powerful tools (Live Search + grok media) are opt-in (issue #168), so this opts them
    into his overlay — exactly how the capital provisions Eddie's tool-set after the migration.
    """
    monkeypatch.setenv("BASECRADLE_TOKEN", FAKE_TOKEN)
    monkeypatch.setenv("BASECRADLE_TIMELINE", TIMELINE_UUID)
    monkeypatch.setenv("AI_PROVIDER", "xai")
    monkeypatch.setenv("AI_MODEL", "grok-4.3")
    monkeypatch.setenv("AI_API_KEY", "xai-test-key")
    monkeypatch.delenv("AI_BASE_URL", raising=False)
    install(
        os.environ["BASECRADLE_CONFIG_HOME"],
        provider="xai",
        opt_in=["xai_search", "grok_generate_image", "grok_generate_video"],
    )
    wire(platform, message_pages=[page(message(uuid=M0, body="hi"))])

    agent = TimelineAgent.from_env()

    assert isinstance(agent.harness.provider, OpenAIProvider)
    assert agent.harness.provider.base_url == "https://api.x.ai/v1"
    # xAI Live Search is wired via search_parameters (web + x), not OpenAI tools entries.
    assert agent.harness.provider._extra_body == {
        "search_parameters": {
            "mode": "on",
            "sources": ["web", "x"],
            "return_citations": True,
        }
    }
    # The grok media tools are active; the OpenAI image tools are not (no OpenAI key/provider).
    names = {t.name for t in agent.harness.tools}
    assert {"grok_generate_image", "grok_generate_video"} <= names
    assert not ({"generate_image", "edit_image", "listen"} & names)


def test_from_env_native_xai_sdk_eddie_keeps_his_opted_in_grok_tools(platform, monkeypatch):
    """Eddie's end state (issue #165): AI_SDK=xai-sdk → the native gRPC brain, his grok tools kept.

    The tool-neutral migration: the *brain* moves to the native SDK, but Eddie's tool-set is his
    own per-persona overlay — opt-in grok tools, unchanged by the SDK swap.
    """
    monkeypatch.setenv("BASECRADLE_TOKEN", FAKE_TOKEN)
    monkeypatch.setenv("BASECRADLE_TIMELINE", TIMELINE_UUID)
    monkeypatch.setenv("AI_PROVIDER", "xai")
    monkeypatch.setenv("AI_SDK", "xai-sdk")
    monkeypatch.setenv("AI_MODEL", "grok-4.3")
    monkeypatch.setenv("AI_API_KEY", "xai-test-key")
    install(
        os.environ["BASECRADLE_CONFIG_HOME"],
        provider="xai",
        opt_in=["grok_generate_image", "grok_generate_video"],
    )
    wire(platform, message_pages=[page(message(uuid=M0, body="hi"))])

    agent = TimelineAgent.from_env()

    assert isinstance(agent.harness.provider, XaiSdkProvider)  # the native gRPC brain
    names = {t.name for t in agent.harness.tools}
    assert {"grok_generate_image", "grok_generate_video"} <= names  # his opted-in tools, kept


def test_from_env_native_xai_sdk_adversarial_persona_resolves_tool_less(platform, monkeypatch):
    """The safety crux (issues #165 + #168): an xai-sdk persona with an empty overlay gets the
    native brain and is NOT armed by the SDK — no powerful tools, no platform tools."""
    monkeypatch.setenv("BASECRADLE_TOKEN", FAKE_TOKEN)
    monkeypatch.setenv("BASECRADLE_TIMELINE", TIMELINE_UUID)
    monkeypatch.setenv("AI_PROVIDER", "xai")
    monkeypatch.setenv("AI_SDK", "xai-sdk")
    monkeypatch.setenv("AI_MODEL", "grok-4.3")
    monkeypatch.setenv("AI_API_KEY", "xai-test-key")
    # Provisioned explicitly tool-less: an installed overlay, emptied (the capital's cutover).
    home = Path(os.environ["BASECRADLE_CONFIG_HOME"])
    install(home, provider="xai")
    for plugin in (home / "tools").glob("*.py"):
        plugin.unlink()
    wire(platform, message_pages=[page(message(uuid=M0, body="hi"))])

    agent = TimelineAgent.from_env()

    assert isinstance(agent.harness.provider, XaiSdkProvider)
    names = {t.name for t in agent.harness.tools}
    # Not armed: no grok media, no Live Search, no platform tools — the SDK granted nothing.
    assert not ({"grok_generate_image", "grok_generate_video"} & names)
    assert not ({"assets", "messages", "timelines", "tasks", "trust", "lock", "delete"} & names)
    assert names <= {"memory"}  # only its private mind remains (itself per-persona configurable)


def test_from_env_wires_the_openai_sdk_provider_by_default(platform, monkeypatch):
    """End to end: the default config gives the agent the openai-SDK Responses brain."""
    monkeypatch.setenv("BASECRADLE_TOKEN", FAKE_TOKEN)
    monkeypatch.setenv("BASECRADLE_TIMELINE", TIMELINE_UUID)
    monkeypatch.setenv("AI_MODEL", "gpt-5.4-mini")
    monkeypatch.setenv("AI_API_KEY", "sk-test-key")
    wire(platform, message_pages=[page(message(uuid=M0, body="hi"))])

    agent = TimelineAgent.from_env()

    assert isinstance(agent.harness.provider, OpenAIProvider)
    assert agent.harness.provider.surface == "responses"


def test_from_env_unlocked_profile_admits_the_opted_in_shell(platform, monkeypatch):
    """End to end (issue #256): `HARNESS_PROFILE=unlocked` builds the registry on the unlocked
    profile, so a persona's opted-in shell actually loads — the poll path wires the same
    `_profile_from_env` decision into `Harness(policy=…)` that the wake path does."""
    monkeypatch.setattr(os, "geteuid", lambda: 1000, raising=False)  # deterministic non-root
    monkeypatch.setenv("BASECRADLE_TOKEN", FAKE_TOKEN)
    monkeypatch.setenv("BASECRADLE_TIMELINE", TIMELINE_UUID)
    monkeypatch.setenv("AI_MODEL", "gpt-5.4-mini")
    monkeypatch.setenv("AI_API_KEY", "sk-test-key")
    monkeypatch.setenv("HARNESS_PROFILE", "unlocked")
    install(os.environ["BASECRADLE_CONFIG_HOME"], provider="openai", opt_in=["shell"])
    wire(platform, message_pages=[page(message(uuid=M0, body="hi"))])

    agent = TimelineAgent.from_env()

    assert "shell" in {t.name for t in agent.harness.tools}


def test_from_env_locked_profile_filters_the_opted_in_shell(platform, monkeypatch):
    """The safe default holds: the same opted-in shell but no `HARNESS_PROFILE` (locked) never
    reaches the registry — it self-excludes at the policy filter rather than crashing construction."""
    monkeypatch.setattr(os, "geteuid", lambda: 1000, raising=False)
    monkeypatch.setenv("BASECRADLE_TOKEN", FAKE_TOKEN)
    monkeypatch.setenv("BASECRADLE_TIMELINE", TIMELINE_UUID)
    monkeypatch.setenv("AI_MODEL", "gpt-5.4-mini")
    monkeypatch.setenv("AI_API_KEY", "sk-test-key")
    monkeypatch.delenv("HARNESS_PROFILE", raising=False)
    install(os.environ["BASECRADLE_CONFIG_HOME"], provider="openai", opt_in=["shell"])
    wire(platform, message_pages=[page(message(uuid=M0, body="hi"))])

    agent = TimelineAgent.from_env()

    assert "shell" not in {t.name for t in agent.harness.tools}


# --- credential bootstrap (mint a token from email + password) ---------------


@pytest.fixture
def no_credentials(monkeypatch):
    """A clean slate: none of the three credential env vars are set."""
    for var in ("BASECRADLE_TOKEN", "BASECRADLE_EMAIL", "BASECRADLE_PASSWORD"):
        monkeypatch.delenv(var, raising=False)


def test_client_from_env_prefers_the_token(no_credentials, monkeypatch, platform):
    """With a token set, that path wins and login is never attempted."""
    monkeypatch.setenv("BASECRADLE_TOKEN", FAKE_TOKEN)
    monkeypatch.setenv("BASECRADLE_EMAIL", "nova@example.com")  # present but ignored
    monkeypatch.setenv("BASECRADLE_PASSWORD", "hunter2-not-used")
    login = platform.post("/session").mock(return_value=httpx.Response(201, json={}))

    client = _client_from_env()

    assert client.token == FAKE_TOKEN
    assert not login.called  # the password path was never taken


def test_client_from_env_mints_a_token_from_credentials(
    no_credentials, monkeypatch, platform, tmp_path
):
    """No token, but credentials → login mints one; that token is carried *and persisted*."""
    monkeypatch.setenv("BASECRADLE_EMAIL", "nova@example.com")
    monkeypatch.setenv("BASECRADLE_PASSWORD", "correct-horse-battery-staple")
    monkeypatch.setenv("BASECRADLE_SESSION_NAME", "nova-harness")
    env = tmp_path / "agent.env"
    env.write_text(
        "BASECRADLE_EMAIL=nova@example.com\nBASECRADLE_PASSWORD=correct-horse-battery-staple\n"
    )
    monkeypatch.setenv("BASECRADLE_ENV_FILE", str(env))
    login = platform.post("/session").mock(
        return_value=httpx.Response(201, json={"token": MINTED_TOKEN, "start_here": None})
    )

    client = _client_from_env()

    assert client.token == MINTED_TOKEN
    assert login.called
    sent = json.loads(login.calls.last.request.content)
    assert sent == {
        "email_address": "nova@example.com",
        "password": "correct-horse-battery-staple",
        "name": "nova-harness",
    }
    # The minted token is written back to the env file so the next wake reuses it.
    assert f"BASECRADLE_TOKEN={MINTED_TOKEN}" in env.read_text()


def test_client_from_env_requires_token_or_credentials(no_credentials):
    """Neither a token nor a full credential pair → a clear error naming both paths."""
    with pytest.raises(ValueError, match="BASECRADLE_TOKEN.*BASECRADLE_EMAIL"):
        _client_from_env()


def test_client_from_env_needs_both_email_and_password(no_credentials, monkeypatch):
    """A half credential (email only) is not enough — it must not silently mint."""
    monkeypatch.setenv("BASECRADLE_EMAIL", "nova@example.com")  # no password
    with pytest.raises(ValueError, match="BASECRADLE_EMAIL"):
        _client_from_env()


def test_from_env_bootstraps_from_credentials_end_to_end(no_credentials, monkeypatch, platform):
    """from_env mints a token and the resulting agent talks to the platform with it."""
    monkeypatch.setenv("BASECRADLE_EMAIL", "nova@example.com")
    monkeypatch.setenv("BASECRADLE_PASSWORD", "correct-horse-battery-staple")
    monkeypatch.setenv("BASECRADLE_TIMELINE", TIMELINE_UUID)
    monkeypatch.setenv("AI_MODEL", "gpt-4o")
    monkeypatch.setenv("AI_API_KEY", "sk-test-key")
    platform.post("/session").mock(
        return_value=httpx.Response(201, json={"token": MINTED_TOKEN, "start_here": None})
    )
    wire(platform, message_pages=[page(message(uuid=M0, body="hi"))])

    agent = TimelineAgent.from_env()

    assert agent.client.token == MINTED_TOKEN
    assert agent.me_uuid == NOVA_UUID
    # The dashboard fetch rode the minted token — proof the bootstrap is live, not just built.
    assert platform.get("/users/dashboard").calls.last.request.headers["Authorization"] == (
        f"Bearer {MINTED_TOKEN}"
    )


# --- wake-on-dashboard onboarding --------------------------------------------


def test_onboarding_seeds_dashboard_orientation_into_the_charter(platform):
    """A fresh peer wakes already knowing what BaseCradle is and where the docs live."""
    wire(
        platform,
        message_pages=[page(message(uuid=M0, body="hi"))],
        dashboard_payload=full_dashboard(),
    )
    agent, _ = build_agent()  # onboard defaults on

    charter = agent.harness.system_prompt
    assert charter is not None
    assert "communications platform where humans and AI are peers" in charter  # summary
    assert "a first-class peer with your own timelines" in charter  # you_are
    assert "https://basecradle.com/docs/api" in charter  # a documentation link

    # And it reaches a session as the system turn the agent actually wakes with.
    first_turn = agent.harness.history[0]
    assert first_turn.role == "system"
    assert "Your BaseCradle orientation:" in first_turn.content


def test_onboarding_composes_with_the_operator_prompt(platform):
    """Orientation is prepended to the operator's charter, not a replacement of it."""
    wire(
        platform,
        message_pages=[page(message(uuid=M0, body="hi"))],
        dashboard_payload=full_dashboard(),
    )
    agent, _ = build_agent(system_prompt="You are Nova. Be concise.")

    charter = agent.harness.system_prompt
    # Both are present, orientation first, operator's charter after.
    assert charter.startswith("Your BaseCradle orientation:")
    assert "You are Nova. Be concise." in charter
    assert charter.index("Your BaseCradle orientation:") < charter.index("You are Nova.")


def test_onboarding_disabled_leaves_only_the_operator_charter(platform):
    """`onboard=False` wakes with only the operator's prompt — no Dashboard text."""
    wire(
        platform,
        message_pages=[page(message(uuid=M0, body="hi"))],
        dashboard_payload=full_dashboard(),
    )
    agent, _ = build_agent(system_prompt="You are Nova.", onboard=False)

    assert agent.harness.system_prompt == "You are Nova."


def test_onboarding_is_a_noop_when_the_dashboard_has_no_orientation(platform):
    """An identity-only Dashboard (older API) leaves the charter untouched."""
    wire(platform, message_pages=[page(message(uuid=M0, body="hi"))])  # identity-only dashboard
    agent, _ = build_agent(system_prompt="You are Nova.")

    assert agent.harness.system_prompt == "You are Nova."  # unchanged, no empty heading


def test_from_env_onboards_by_default(platform, monkeypatch):
    monkeypatch.setenv("BASECRADLE_TOKEN", FAKE_TOKEN)
    monkeypatch.setenv("BASECRADLE_TIMELINE", TIMELINE_UUID)
    monkeypatch.setenv("AI_MODEL", "gpt-4o")
    monkeypatch.setenv("AI_API_KEY", "sk-test-key")
    monkeypatch.delenv("HARNESS_ONBOARD", raising=False)
    monkeypatch.delenv("HARNESS_SYSTEM_PROMPT", raising=False)
    wire(
        platform,
        message_pages=[page(message(uuid=M0, body="hi"))],
        dashboard_payload=full_dashboard(),
    )

    agent = TimelineAgent.from_env()

    assert "Your BaseCradle orientation:" in agent.harness.system_prompt


def test_from_env_onboarding_can_be_disabled(platform, monkeypatch):
    monkeypatch.setenv("BASECRADLE_TOKEN", FAKE_TOKEN)
    monkeypatch.setenv("BASECRADLE_TIMELINE", TIMELINE_UUID)
    monkeypatch.setenv("AI_MODEL", "gpt-4o")
    monkeypatch.setenv("AI_API_KEY", "sk-test-key")
    monkeypatch.setenv("HARNESS_ONBOARD", "0")
    monkeypatch.setenv("HARNESS_SYSTEM_PROMPT", "You are Nova.")
    wire(
        platform,
        message_pages=[page(message(uuid=M0, body="hi"))],
        dashboard_payload=full_dashboard(),
    )

    agent = TimelineAgent.from_env()

    assert agent.harness.system_prompt == "You are Nova."  # onboarding off → no orientation


# --- the orientation/compose/flag helpers, in isolation ----------------------


def test_orientation_renders_environment_and_documentation():
    from basecradle import BaseCradle as _BC

    with respx.mock(base_url=BC_URL, assert_all_called=False) as router:
        router.get("/users/dashboard").mock(return_value=httpx.Response(200, json=full_dashboard()))
        dash = _BC(token=FAKE_TOKEN).me
        text = _orientation(dash)

    assert text.startswith("Your BaseCradle orientation:")
    assert "You are on BaseCradle — a communications platform" in text
    assert "Here, you are a first-class peer" in text
    assert "- API: https://basecradle.com/docs/api" in text
    assert "- Changelog: https://basecradle.com/docs/changelog" in text


def test_orientation_is_none_when_the_dashboard_lacks_orientation():
    from basecradle import BaseCradle as _BC

    with respx.mock(base_url=BC_URL, assert_all_called=False) as router:
        router.get("/users/dashboard").mock(return_value=httpx.Response(200, json=dashboard()))
        dash = _BC(token=FAKE_TOKEN).me
        assert _orientation(dash) is None  # identity-only → nothing to say


def test_compose_prompt_orders_and_handles_absence():
    assert _compose_prompt("ORIENT", "CHARTER") == "ORIENT\n\nCHARTER"
    assert _compose_prompt("ORIENT", None) == "ORIENT"
    assert _compose_prompt(None, "CHARTER") == "CHARTER"
    assert _compose_prompt(None, None) is None


def test_onboard_from_env_defaults_on_and_parses_falsy(monkeypatch):
    monkeypatch.delenv("HARNESS_ONBOARD", raising=False)
    assert _onboard_from_env() is True  # unset → on

    for falsy in ("0", "false", "False", "no", "off", " OFF "):
        monkeypatch.setenv("HARNESS_ONBOARD", falsy)
        assert _onboard_from_env() is False

    # Unset stays on; so does any non-off value, blank included — off only when
    # explicitly turned off.
    for truthy in ("1", "true", "yes", "on", "", "   "):
        monkeypatch.setenv("HARNESS_ONBOARD", truthy)
        assert _onboard_from_env() is True
