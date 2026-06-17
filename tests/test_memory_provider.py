"""The pluggable memory seam: the `MemoryProvider` interface, the default, and the hooks.

Two layers are pinned here:

1. **Unit** — the default `SqliteMemoryProvider` implements tools + store and leaves
   `observe`/`context` as no-ops (the behavior-preserving default), and
   `memory_provider_from_env` selects sqlite by default, a custom class by dotted path,
   and errors clearly on a bad value.
2. **Integration** — a stub provider proves the wake fires `observe` after *each*
   exchange and injects `context` into the persistent Turn-0 brief, against a respx-mocked
   platform. The cast is the fixed fiction: Nova Digital (`nova`, AI) is the agent; John
   Doe (`john`).
"""

import httpx
import pytest
import respx
from basecradle import BaseCradle

from basecradle_harness import (
    Harness,
    MemoryExchange,
    MemoryProvider,
    MemoryScope,
    MemoryTool,
    SqliteMemoryProvider,
    SqliteMemoryStore,
    WakeAgent,
    memory_provider_from_env,
)
from basecradle_harness._messages import Message

# --- the default SQLite provider: tools + store, no-op hooks ------------------


def test_sqlite_provider_supplies_the_memory_tool_over_its_store(tmp_path):
    provider = SqliteMemoryProvider(tmp_path / "memory.db")
    tools = provider.tools()

    assert [type(t) for t in tools] == [MemoryTool]
    assert isinstance(provider.store, SqliteMemoryStore)
    # The tool dispatches onto the provider's one store — a write through the tool is
    # readable straight off the store, so the model's ops and any hook share state.
    tools[0].run(action="write", key="city", value="Dallas")
    assert provider.store.read("city") == "Dallas"
    provider.close()


def test_sqlite_provider_hooks_are_no_ops(tmp_path):
    """The default provider keeps explicit-memory behavior: observe/context do nothing."""
    provider = SqliteMemoryProvider(tmp_path / "memory.db")
    scope = MemoryScope(agent="agent-uuid", timeline="tl-uuid", query="anything")

    assert provider.context(scope) is None
    # observe is a no-op that must not raise and must not write anything.
    provider.observe(MemoryExchange(user="hi", assistant="hello", scope=scope))
    assert provider.store.list() == "No memories stored yet."
    provider.close()


def test_sqlite_provider_close_closes_the_shared_store(tmp_path):
    provider = SqliteMemoryProvider(tmp_path / "memory.db")
    provider.store.write("k", "v")  # opens the connection
    assert provider.store._conn is not None
    provider.close()
    assert provider.store._conn is None


def test_tool_given_a_shared_store_does_not_close_it(tmp_path):
    """A tool sharing a provider's store leaves closing to the provider that owns it."""
    store = SqliteMemoryStore(tmp_path / "memory.db")
    tool = MemoryTool(store=store)
    tool.run(action="write", key="k", value="v")  # opens the connection

    tool.close()  # the tool does not own the store
    assert store._conn is not None  # still open — the provider closes it
    store.close()


# --- provider selection from the environment ---------------------------------


def test_from_env_defaults_to_sqlite(monkeypatch):
    monkeypatch.delenv("HARNESS_MEMORY_PROVIDER", raising=False)
    assert isinstance(memory_provider_from_env(), SqliteMemoryProvider)


def test_from_env_sqlite_explicit_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("HARNESS_MEMORY_PROVIDER", "SQLite")
    assert isinstance(memory_provider_from_env(), SqliteMemoryProvider)


def test_from_env_home_points_the_store_at_the_agent_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HARNESS_MEMORY_PROVIDER", "sqlite")
    provider = memory_provider_from_env(home=tmp_path)
    assert provider.store.path == tmp_path / "memory.db"


def test_from_env_loads_a_custom_provider_by_dotted_path(monkeypatch):
    """A 'module:Class' value imports any MemoryProvider subclass — the custom-provider seam."""
    monkeypatch.setenv(
        "HARNESS_MEMORY_PROVIDER", "basecradle_harness._memory_provider:SqliteMemoryProvider"
    )
    assert isinstance(memory_provider_from_env(), SqliteMemoryProvider)


def test_from_env_rejects_a_meaningless_value(monkeypatch):
    monkeypatch.setenv("HARNESS_MEMORY_PROVIDER", "nodots")
    with pytest.raises(ValueError, match="not a known provider"):
        memory_provider_from_env()


def test_from_env_rejects_a_non_provider_target(monkeypatch):
    monkeypatch.setenv("HARNESS_MEMORY_PROVIDER", "json:dumps")
    with pytest.raises(TypeError, match="not a MemoryProvider"):
        memory_provider_from_env()


def test_from_env_reports_an_unimportable_path(monkeypatch):
    monkeypatch.setenv("HARNESS_MEMORY_PROVIDER", "no_such_module:Thing")
    with pytest.raises(ValueError, match="Could not import"):
        memory_provider_from_env()


# --- a stub provider, wired through a real wake -------------------------------

BC_URL = "https://basecradle.com"
FAKE_TOKEN = "bc_uat_KqI8zFxkQ0OZ8vYwT7mWcVtR3nSdLpEa"
NOVA_UUID = "019e7750-66ee-79c8-ad8a-bbb6ea7c2bcc"  # the agent (me)
JOHN_UUID = "019e7750-66ee-7e50-9e54-3bf8c3d6a8f1"  # the human
TIMELINE_UUID = "019e7750-66ee-7f53-829f-13a8a710b6da"
MSELF = "019e7750-9a9a-7b7b-8c8c-0a0b0c0d0e0f"  # the agent's own prior post (oldest)
M0 = "019e7751-4a1b-7c2d-8e3f-1a2b3c4d5e6f"
M1 = "019e7752-5b2c-7d3e-9f40-2b3c4d5e6f70"
REPLY = "019e7755-8e5f-7f70-9283-5e6f70819203"


def _message(*, uuid, body, mine=False):
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


def _dashboard():
    return {"identity": {"uuid": NOVA_UUID, "handle": "nova", "name": "Nova Digital", "kind": "ai"}}


def _timeline():
    return {
        "timeline": {
            "uuid": TIMELINE_UUID,
            "name": "Test",
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


class RecordingProvider(MemoryProvider):
    """A stub provider that records every observed exchange and injects a fixed context."""

    INJECTED = "Recalled: John lives in Dallas."

    def __init__(self):
        self.observed: list[MemoryExchange] = []
        self.context_scopes: list[MemoryScope] = []

    def observe(self, exchange):
        self.observed.append(exchange)

    def context(self, scope):
        self.context_scopes.append(scope)
        return self.INJECTED


class _CannedModel:
    def __init__(self, text="Hello, John."):
        self.text = text

    def chat(self, messages, tools=None):
        return Message.assistant(content=self.text)


@pytest.fixture
def platform():
    with respx.mock(base_url=BC_URL, assert_all_called=False) as router:
        router.get("/users/dashboard").mock(return_value=httpx.Response(200, json=_dashboard()))
        router.get("/users/dashboard.md").mock(
            return_value=httpx.Response(200, text="# Dashboard\n\nWelcome.\n")
        )
        router.get(f"/timelines/{TIMELINE_UUID}").mock(
            return_value=httpx.Response(200, json=_timeline())
        )
        router.post(f"/timelines/{TIMELINE_UUID}/messages").mock(
            return_value=httpx.Response(
                201, json={"message": _message(uuid=REPLY, body="reply", mine=True)}
            )
        )
        router.get("/assets").mock(
            return_value=httpx.Response(200, json={"assets": [], "next_cursor": None})
        )
        router.get("/webhook_events").mock(
            return_value=httpx.Response(200, json={"webhook_events": [], "next_cursor": None})
        )
        router.get("/tasks").mock(
            return_value=httpx.Response(200, json={"tasks": [], "next_cursor": None})
        )
        yield router


def _serve_messages(platform, *pages):
    platform.get("/messages").mock(side_effect=[httpx.Response(200, json=p) for p in pages])


def _wake(home, memory_provider):
    client = BaseCradle(token=FAKE_TOKEN)
    harness = Harness(_CannedModel(), home=home)
    return WakeAgent(
        harness,
        timeline=TIMELINE_UUID,
        client=client,
        onboard=True,
        memory_provider=memory_provider,
    )


def test_observe_fires_after_each_exchange(platform, tmp_path):
    """Two unseen peer messages → the provider observes two exchanges, user + reply intact.

    A prior self-post is the oldest message, so the first-wake split replies to *everything*
    after it (both peer messages), not just the newest — exercising observe across a batch.
    """
    _serve_messages(
        platform,
        {
            "messages": [
                _message(uuid=M1, body="Second?"),
                _message(uuid=M0, body="First?"),
                _message(uuid=MSELF, body="earlier", mine=True),
            ],
            "next_cursor": None,
        },
    )
    provider = RecordingProvider()
    agent = _wake(tmp_path, provider)

    agent.wake()

    assert len(provider.observed) == 2
    first, second = provider.observed
    assert "First?" in first.user and first.assistant == "Hello, John."
    assert "Second?" in second.user
    # Scope is the agent identity, with the timeline as metadata (cross-timeline memory).
    assert first.scope.agent == NOVA_UUID
    assert first.scope.timeline == TIMELINE_UUID


def test_context_is_injected_into_the_turn0_brief(platform, tmp_path):
    _serve_messages(
        platform, {"messages": [_message(uuid=M0, body="Where do I live?")], "next_cursor": None}
    )
    provider = RecordingProvider()
    agent = _wake(tmp_path, provider)

    agent.wake()

    # The recalled context rode into the persistent brief as a system turn.
    history = agent.harness.session(agent.source).history
    briefs = [
        m.content
        for m in history
        if m.role == "system" and RecordingProvider.INJECTED in (m.content or "")
    ]
    assert briefs, "the provider's context() output should be in the Turn-0 brief"
    # The hook was scoped to the agent and handed the incoming turn as the retrieval query.
    assert provider.context_scopes[0].agent == NOVA_UUID
    assert provider.context_scopes[0].query == "john: Where do I live?"


def test_a_self_skip_does_not_observe(platform, tmp_path):
    """The agent's own post is skipped — no exchange, so observe never fires for it."""
    _serve_messages(
        platform,
        {"messages": [_message(uuid=M0, body="my own post", mine=True)], "next_cursor": None},
    )
    provider = RecordingProvider()
    agent = _wake(tmp_path, provider)

    agent.wake()

    assert provider.observed == []


def test_a_raising_hook_never_breaks_the_wake(platform, tmp_path):
    """observe/context that raise degrade gracefully — the reply still posts."""

    class BrokenProvider(MemoryProvider):
        def observe(self, exchange):
            raise RuntimeError("observe boom")

        def context(self, scope):
            raise RuntimeError("context boom")

    _serve_messages(platform, {"messages": [_message(uuid=M0, body="hi")], "next_cursor": None})
    agent = _wake(tmp_path, BrokenProvider())

    posted = agent.wake()  # must not raise

    assert len(posted) == 1  # the reply landed despite both hooks raising
