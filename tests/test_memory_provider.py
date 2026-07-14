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

from importlib import metadata

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
from basecradle_harness._memory_provider import describe_memory_provider
from basecradle_harness._mempalace import MemPalaceMemoryProvider
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


# --- the bound provider's manifest identity (issue #269) ----------------------
#
# `--resolved-config` reports the memory axis off these, so what they pin is that the answer
# comes from the provider **object that was bound**, never from a re-read of the env var — the
# whole point being that a dropped `HARNESS_MEMORY_PROVIDER` must be *visible* as a silent
# fallback to SQLite, not reported as whatever the introspecting shell happened to be told.


class _CustomProvider(MemoryProvider):
    """A custom provider: not a built-in, so it names itself by import path."""


class _TunedSqliteProvider(SqliteMemoryProvider):
    """A *subclass* of a built-in — a custom provider, and reported as one."""


def test_describe_names_the_default_sqlite_provider_with_no_backing_package(tmp_path):
    """`sqlite` ships inside the harness (stdlib sqlite3), so it has no separately-pinned
    package: the version is `None`, not a second copy of `harness_version`."""
    assert describe_memory_provider(SqliteMemoryProvider(tmp_path / "memory.db")) == (
        "sqlite",
        None,
    )


def test_describe_names_a_bound_mempalace_provider_whose_extra_is_missing(tmp_path):
    """Bound to MemPalace with the extra *not installed* → `("mempalace", None)`.

    A real, reachable state: binding is lazy (the adapter imports MemPalace only on the first
    `observe`/`context`), so an agent selects the provider fine and then loses its memory at the
    first wake. `None` on the version of a *named* backing package is the defect signal that
    catches it off-box — the case the test env genuinely reproduces, since MemPalace is an
    optional extra this suite never installs.
    """
    assert describe_memory_provider(MemPalaceMemoryProvider(tmp_path / "palace")) == (
        "mempalace",
        None,
    )


def test_describe_reports_the_mempalace_extras_installed_version(tmp_path, monkeypatch):
    """With the extra installed, the version is ground truth read off its distribution — the
    field basecradle-noc#195's `pinned_extra_versions` drift axis compares against the pin.
    MemPalace is not installed here, so its distribution metadata is the one thing faked."""
    monkeypatch.setattr(metadata, "version", lambda dist: {"mempalace": "3.5.0"}[dist])

    assert describe_memory_provider(MemPalaceMemoryProvider(tmp_path / "palace")) == (
        "mempalace",
        "3.5.0",
    )


def test_describe_reads_the_bound_class_not_the_env_var(monkeypatch):
    """A dotted path naming a *built-in* class reports the built-in's alias: the bound object is
    the truth, so the two spellings of one provider can never report as two different ones."""
    monkeypatch.setenv(
        "HARNESS_MEMORY_PROVIDER", "basecradle_harness._memory_provider:SqliteMemoryProvider"
    )
    assert describe_memory_provider(memory_provider_from_env())[0] == "sqlite"


def test_describe_names_a_custom_provider_by_its_import_path():
    """A custom provider reports the `module:Class` actually bound — including a subclass of a
    built-in, which *is* a custom provider and must not hide behind the built-in's alias.

    Its version is `None`: the harness pins no package for someone else's class, and a module's
    name is not its distribution's — so there is nothing here it can honestly report.
    """
    assert describe_memory_provider(_CustomProvider()) == (
        "tests.test_memory_provider:_CustomProvider",
        None,
    )
    assert describe_memory_provider(_TunedSqliteProvider())[0] == (
        "tests.test_memory_provider:_TunedSqliteProvider"
    )


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
        #: The turns the model was shown on its last call — the only place the ephemeral
        #: per-wake brief is observable, since it is never written to the transcript (#275).
        self.shown: list[Message] = []

    def chat(self, messages, tools=None):
        self.shown = list(messages)
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
    """Serve the newest-first message list; the LAST page repeats for every read.

    A #226 message wake reads the list several times per turn (initial gather + Loop-1 settle +
    Loop-2 staleness re-checks), so the given pages are served once in order and the last one
    repeats — a re-read after the mark advanced yields nothing newer.
    """
    queue = [httpx.Response(200, json=p) for p in pages]

    def _serve(request):
        return queue.pop(0) if len(queue) > 1 else queue[0]

    platform.get("/messages").mock(side_effect=_serve)


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


def test_observe_fires_once_for_a_batched_exchange(platform, tmp_path):
    """Two unseen peer messages → ONE batched exchange → the provider observes it once (#226).

    A prior self-post is the oldest message, so the first-wake split replies to *everything*
    after it (both peer messages). Under the #226 many-to-one batch reply that is a single
    exchange — both incoming messages seeded as one turn, one reply — so `observe` fires once,
    with both messages in `user` and the single reply in `assistant`.
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

    assert len(provider.observed) == 1  # one exchange for the whole batch
    (exchange,) = provider.observed
    assert "First?" in exchange.user and "Second?" in exchange.user  # both, oldest-first
    assert exchange.assistant == "Hello, John."  # the single batched reply
    # Scope is the agent identity, with the timeline as metadata (cross-timeline memory).
    assert exchange.scope.agent == NOVA_UUID
    assert exchange.scope.timeline == TIMELINE_UUID


def test_context_is_injected_into_the_turn0_brief(platform, tmp_path):
    """Recalled memory reaches the *model* inside the per-wake brief — and, like the rest of
    the brief, is never written to the transcript (issue #275): it is a fresh retrieval each
    wake, so a persisted copy would be a stale answer to an old question, re-billed forever."""
    _serve_messages(
        platform, {"messages": [_message(uuid=M0, body="Where do I live?")], "next_cursor": None}
    )
    provider = RecordingProvider()
    agent = _wake(tmp_path, provider)

    agent.wake()

    model = agent.harness.provider
    shown = [m for m in model.shown if RecordingProvider.INJECTED in (m.content or "")]
    assert shown, "the provider's context() output should ride the brief the model was shown"
    history = agent.harness.session(agent.source).history
    assert not [m for m in history if RecordingProvider.INJECTED in (m.content or "")]
    # The hook was scoped to the agent and handed the incoming turn as the retrieval query.
    assert provider.context_scopes[0].agent == NOVA_UUID
    assert provider.context_scopes[0].query == "[2026-06-04T00:00:00.000Z] john: Where do I live?"


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


def test_memory_observes_a_silent_turn(platform, tmp_path):
    """A peer states a fact, the agent stays silent — and memory keeps it anyway (issue #293).

    **The founder's acceptance test for the Unspoken Channel**, and the reason `_observe` is now
    unconditional. Under silence-default the agent posts nothing on most turns; the old guard
    (`if reply.strip()` — observe only when a reply posted) would therefore have made memory *stop
    seeing* the very messages the agent chose not to answer. That inverts what memory is for: "my
    birthday is Feb. 16, 1977" is worth remembering precisely when nobody needed to reply to it.

    So the trigger and the (unspoken) narration are handed to the provider on every **engaged**
    turn, spoken or not — which is what makes the fact recallable from another timeline later.
    """
    _serve_messages(
        platform,
        {
            "messages": [_message(uuid=M0, body="My birthday is Feb. 16, 1977.")],
            "next_cursor": None,
        },
    )
    provider = RecordingProvider()
    client = BaseCradle(token=FAKE_TOKEN)
    # A model that says nothing to the timeline (no tools, so nothing *can* be posted) and ends its
    # turn with a plain thought — the shape of an ordinary silent turn.
    harness = Harness(_CannedModel(text="Noted — no reply needed."), home=tmp_path)
    agent = WakeAgent(
        harness, timeline=TIMELINE_UUID, client=client, onboard=True, memory_provider=provider
    )

    posted = agent.wake()

    assert posted == []  # silence: nothing reached the timeline
    assert len(provider.observed) == 1  # but the exchange was still observed
    exchange = provider.observed[0]
    assert "My birthday is Feb. 16, 1977." in exchange.user
    assert exchange.assistant == "Noted — no reply needed."  # the narration, not a posted reply


def test_a_raising_hook_never_breaks_the_wake(platform, tmp_path):
    """observe/context that raise degrade gracefully — the wake still completes."""

    class BrokenProvider(MemoryProvider):
        def observe(self, exchange):
            raise RuntimeError("observe boom")

        def context(self, scope):
            raise RuntimeError("context boom")

    _serve_messages(platform, {"messages": [_message(uuid=M0, body="hi")], "next_cursor": None})
    agent = _wake(tmp_path, BrokenProvider())

    posted = agent.wake()  # must not raise

    assert posted == []  # the model called no tools, so it said nothing — and that is fine
    # The wake ran to completion despite both hooks raising: the message is marked seen, so it is
    # never re-read. (A memory backend hiccup must cost the memory, never the turn.)
    assert agent.marks.get(TIMELINE_UUID) == M0


# === Compaction summaries reach durable memory (issue #276, requirement 7) ===
#
# `observe` is handed the *dialogue* only — which is right, and is what keeps the palace worth
# searching. But it means tool-driven work leaves no durable trace unless the agent narrated it.
# That is harmless while the turns are still in the transcript, and stops being harmless the moment
# compaction drops them. So the boundary is where the work is captured.


def test_a_compaction_summary_is_written_to_a_providers_store(platform, tmp_path):
    """The default SQLite provider: `observe` is a no-op, so the summary goes to the store."""
    memory = SqliteMemoryProvider(tmp_path / "memory.db")
    agent = _wake(tmp_path, memory)

    agent._remember_compaction("WORK DONE: I posted the weekly report, asset 0198…")

    stored = memory.store.list()
    assert "compaction/timeline:" in stored
    # The agent can read its own past back — nothing is dropped from a transcript without a record.
    key = next(k for k in stored.splitlines() if "compaction/" in k).strip("- ").strip()
    assert "I posted the weekly report" in memory.store.read(key)
    memory.close()


def test_a_storeless_provider_gets_the_summary_through_observe(platform, tmp_path):
    """MemPalace is a pure middleware: its `store` is None by design, so `observe` is the surface."""
    memory = RecordingProvider()
    memory.store = None
    agent = _wake(tmp_path, memory)

    agent._remember_compaction("WORK DONE: I filed the issue.")

    assert len(memory.observed) == 1
    exchange = memory.observed[0]
    assert exchange.assistant == "WORK DONE: I filed the issue."
    assert "compaction" in exchange.user.lower()  # framed as the agent's own notes, not a peer's
    # Scoped to the *agent*, never partitioned by timeline — memory is one mind across every channel.
    assert exchange.scope.agent == agent.me_uuid


def test_a_memory_failure_never_undoes_a_compaction(platform, tmp_path):
    class Exploding(RecordingProvider):
        store = None

        def observe(self, exchange):
            raise RuntimeError("the palace is on fire")

    agent = _wake(tmp_path, Exploding())
    compactor = agent.harness.compactor

    # The wake wires the hook; the compactor guards it. Memory is best-effort — the transcript
    # bound is not, and a failed write must never leave a transcript uncompacted.
    if compactor is not None:
        compactor.on_summary = agent._remember_compaction
        compactor._remember("SUMMARY")  # must not raise
