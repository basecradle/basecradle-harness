"""The mining boundary: only dialogue reaches a mining memory provider (issue #438).

The founder's invariant, in one sentence: a mining provider stores **the text sent into the
harness** and **the LLM's output**, and nothing the harness composed. The observed violation
was two of five Turn-0 recall hits on @briggs being copies of the *pre-0.112.0 recall heading
itself* — the brief, mined, and served back to him as memory.

These tests are the enforcement. The first one is the proof the issue asked for: a synthetic
wake whose every harness-composed surface carries a distinct sentinel, run end to end, with the
assertion that the model **saw** each sentinel and the memory provider was handed **none** of
them. It fails if any future edit re-opens a path, without anyone having to think of that path.

The cast is the fixed fiction: Nova Digital (`nova`, AI) is the agent; John Doe (`john`).
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
    WakeAgent,
    _mining,
)
from basecradle_harness import _wake as wake_module
from basecradle_harness._engine import EngineError
from basecradle_harness._install import install
from basecradle_harness._mempalace import _CLOSE_TAG, _INJECTED_HEADING, _OPEN_TAG
from basecradle_harness._messages import Message
from basecradle_harness._mining import _LEGACY_RECALL_HEADING as LEGACY_HEADING
from basecradle_harness._mining import (
    MIN_UNIT_CHARS,
    Verdict,
    catalog,
    classify,
    strip_injected,
)

BC_URL = "https://basecradle.com"
FAKE_TOKEN = "bc_uat_KqI8zFxkQ0OZ8vYwT7mWcVtR3nSdLpEa"

NOVA_UUID = "019e7740-0000-7000-8000-000000000001"
JOHN_UUID = "019e7740-0000-7000-8000-000000000002"
TIMELINE_UUID = "019e7750-66ee-7f53-829f-13a8a710b6da"
M0 = "019e7751-4a1b-7c2d-8e3f-1a2b3c4d5e6f"
MSELF = "019e7750-9a9a-7b7b-8c8c-0a0b0c0d0e0f"
REPLY = "019e7755-8e5f-7f70-9283-5e6f70819203"
ASSET_UUID = "019e7760-1111-7222-8333-444455556666"
TASK_UUID = "019e7761-2222-7333-8444-555566667777"
EVENT_UUID = "019e7762-3333-7444-8555-666677778888"
ENDPOINT_UUID = "019e7763-4444-7555-8666-777788889999"

# One sentinel per harness-composed surface. Each is a long, unmistakable string, because the
# assertion is a substring search over what the provider was handed — and a short sentinel could
# be a coincidence in ordinary conversation.
CHARTER_SENTINEL = "SENTINEL-CHARTER-do-not-mine-this-personality-charter-line"
MANIFEST_SENTINEL = "SENTINEL-MANIFEST-do-not-mine-this-tool-note"
DASHBOARD_SENTINEL = "SENTINEL-DASHBOARD-do-not-mine-this-platform-primer"
RECALL_SENTINEL = "SENTINEL-RECALL-do-not-mine-this-recalled-memory"
SENTINELS = (CHARTER_SENTINEL, MANIFEST_SENTINEL, DASHBOARD_SENTINEL, RECALL_SENTINEL)


class Recorder(MemoryProvider):
    """Records what it is asked to mine, and injects a sentinel into the Turn-0 brief."""

    def __init__(self, injected=RECALL_SENTINEL):
        self.observed: list[MemoryExchange] = []
        self.injected = injected

    def observe(self, exchange):
        self.observed.append(exchange)

    def context(self, scope):
        return self.injected


class _CannedModel:
    """A model that answers with fixed text, and remembers everything it was shown."""

    def __init__(self, text="Hello, John.", raises=False):
        self.text = text
        self.raises = raises
        self.shown: list[Message] = []

    def chat(self, messages, tools=None):
        self.shown = list(messages)
        if self.raises:
            raise EngineError("the reserve summary failed too")
        return Message.assistant(content=self.text)


def _message(*, uuid, body, mine=False):
    who = (
        {"uuid": NOVA_UUID, "handle": "nova", "name": "Nova Digital", "kind": "ai"}
        if mine
        else {"uuid": JOHN_UUID, "handle": "john", "name": "John Doe", "kind": "human"}
    )
    return {
        "uuid": f"item-{uuid}",
        "created_at": "2026-08-29T12:00:00Z",
        "user": who,
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


@pytest.fixture
def platform():
    with respx.mock(base_url=BC_URL, assert_all_called=False) as router:
        router.get("/users/dashboard").mock(return_value=httpx.Response(200, json=_dashboard()))
        router.get("/users/dashboard.md").mock(
            return_value=httpx.Response(200, text=f"# Dashboard\n\n{DASHBOARD_SENTINEL}\n")
        )
        router.get(f"/timelines/{TIMELINE_UUID}").mock(
            return_value=httpx.Response(200, json=_timeline())
        )
        router.post(f"/timelines/{TIMELINE_UUID}/messages").mock(
            return_value=httpx.Response(
                201, json={"message": _message(uuid=REPLY, body="reply", mine=True)}
            )
        )
        router.get("/messages").mock(
            return_value=httpx.Response(
                200,
                json={
                    "messages": [
                        _message(uuid=M0, body="Where does John live?"),
                        _message(uuid=MSELF, body="earlier", mine=True),
                    ],
                    "next_cursor": None,
                },
            )
        )
        for path in ("/assets", "/webhook_events", "/tasks"):
            key = path.strip("/")
            router.get(path).mock(
                return_value=httpx.Response(200, json={key: [], "next_cursor": None})
            )
        yield router


def _agent(home, provider, model=None, monkeypatch=None):
    if monkeypatch is not None:
        monkeypatch.setenv("HARNESS_SYSTEM_PROMPT", f"You are Nova.\n\n{CHARTER_SENTINEL}")
    harness = Harness(model or _CannedModel(), home=home)
    return WakeAgent(
        harness,
        timeline=TIMELINE_UUID,
        client=BaseCradle(token=FAKE_TOKEN),
        onboard=True,
        memory_provider=provider,
        tool_manifest=[("memory", MANIFEST_SENTINEL)],
    )


# === the proof: every harness-composed surface is shown, and none of it is mined ===============


def test_no_part_of_the_brief_reaches_the_mined_exchange(platform, tmp_path, monkeypatch):
    """The issue's acceptance test: sentinels in charter, manifest, dashboard and recall.

    Both halves matter and both are asserted. If the sentinels never reached the *model*, this
    would pass for the wrong reason — a brief that composed empty proves nothing about a
    boundary. So the model's own view is checked first, and only then the provider's.
    """
    provider = Recorder()
    model = _CannedModel(text="John lives in Dallas.")
    agent = _agent(tmp_path, provider, model, monkeypatch)

    agent.wake()

    shown = "\n".join(m.content or "" for m in model.shown)
    for sentinel in SENTINELS:
        assert sentinel in shown, f"{sentinel} never reached the model — the test proves nothing"
    assert provider.observed, "the turn was never mined at all"
    mined = "\n".join(e.user + "\n" + e.assistant for e in provider.observed)
    for sentinel in SENTINELS:
        assert sentinel not in mined
    # And what *is* mined is the dialogue: the peer's message and the model's own words.
    assert "Where does John live?" in mined
    assert "John lives in Dallas." in mined


def test_a_canned_stuck_note_is_never_mined_as_the_agents_words(platform, tmp_path, monkeypatch):
    """A degraded turn's narration is the harness's sentence, not the model's (`_STUCK_NOTE`).

    The peer's half is still mined: their message is real whatever happened on our side.
    """
    provider = Recorder()
    agent = _agent(tmp_path, provider, _CannedModel(raises=True), monkeypatch)

    agent.wake()

    assert provider.observed
    exchange = provider.observed[-1]
    assert exchange.assistant == ""
    assert wake_module._STUCK_NOTE not in exchange.user
    assert "Where does John live?" in exchange.user


def test_a_reply_quoting_the_recall_block_is_stripped_of_its_framing(platform, tmp_path):
    """Defense in depth: the model quoting its own injected recall does not re-file it.

    The one residue closure cannot reach — it arrives as genuine LLM output on a path that is
    genuinely mined — so it is removed here, and only it. The real words survive.
    """
    provider = Recorder(injected=None)
    quoted = f"{_INJECTED_HEADING}\n{_OPEN_TAG}\n- old fact\n{_CLOSE_TAG}\nJohn lives in Dallas."
    agent = _agent(tmp_path, provider, _CannedModel(text=quoted))

    agent.wake()

    assistant = provider.observed[-1].assistant
    assert "John lives in Dallas." in assistant
    assert "old fact" in assistant  # a recalled fact the model restated is its own text now
    for literal in (_INJECTED_HEADING, _OPEN_TAG, _CLOSE_TAG):
        assert literal not in assistant


def test_an_exchange_that_is_nothing_but_scaffolding_is_not_mined_at_all(platform, tmp_path):
    """Stripping can empty an exchange, and an empty drawer only ever dilutes retrieval."""
    provider = Recorder()
    agent = _agent(tmp_path, provider)

    agent._observe(f"{LEGACY_HEADING}\n", _OPEN_TAG)

    assert provider.observed == []


# === the item renderings: framing is for the model, content is for the palace ==================


class _Content:
    def __init__(self, **fields):
        self.__dict__.update(fields)


def _asset():
    return _Content(
        created_at="2026-08-29T12:00:00Z",
        user=_Content(handle="john"),
        content=_Content(
            uuid=ASSET_UUID,
            file=_Content(filename="skyline.png", content_type="image/png", byte_size=1234),
        ),
    )


def _task():
    return _Content(
        created_at="2026-08-29T12:00:00Z",
        content=_Content(
            uuid=TASK_UUID,
            activate_at="2026-08-29T12:00:00Z",
            instructions="Post the weekly report.",
        ),
    )


def _event():
    return _Content(
        created_at="2026-08-29T12:00:00Z",
        webhook_endpoint=_Content(uuid=ENDPOINT_UUID),
        content=_Content(uuid=EVENT_UUID, content_type="application/json", payload='{"ok":true}'),
    )


@pytest.mark.parametrize(
    ("item", "kind", "framing", "content"),
    [
        (_asset(), wake_module._ASSETS, "Use the assets tool", "skyline.png"),
        (_task(), wake_module._TASKS, "Carry out its instructions", "Post the weekly report."),
        (_event(), wake_module._EVENTS, "Decide whether and how", '{"ok":true}'),
    ],
)
def test_an_items_dialogue_keeps_its_content_and_drops_the_harness_instruction(
    item, kind, framing, content
):
    """`_dialogue_of` is the mined rendering; the model-facing one still carries the framing.

    Both assertions are the point: dropping the instruction from what the *model* reads would be
    a different (and much worse) bug than mining it.
    """
    dialogue = wake_module._dialogue_of(item, kind)

    assert content in dialogue
    assert "2026-08-29T12:00:00Z" in dialogue  # provenance stays: it is what recall searches on
    assert framing not in dialogue

    shown = {
        wake_module._ASSETS: wake_module._incoming_asset_text,
        wake_module._TASKS: wake_module._activated_task_text,
        wake_module._EVENTS: wake_module._incoming_event_text,
    }[kind](item)
    assert framing in shown
    assert content in shown


def test_a_peers_message_is_mined_with_its_provenance(tmp_path):
    """The message path was already dialogue, and stays byte-identical — handle and stamp kept."""
    message = _message(uuid=M0, body="Where does John live?")
    item = _Content(
        created_at=message["created_at"],
        user=_Content(handle="john"),
        content=_Content(uuid=M0, body="Where does John live?"),
    )

    assert wake_module._dialogue_of(item, wake_module._MESSAGES) == (
        "[2026-08-29T12:00:00Z] john: Where does John live?"
    )


# === strip_injected ============================================================================


def test_strip_injected_is_case_insensitive_and_removes_only_the_recall_framing():
    text = f"{LEGACY_HEADING.upper()}\n<MEMPALACE-RECALL>\nJohn lives in Dallas.\n{_CLOSE_TAG}"

    stripped = strip_injected(text)

    assert "John lives in Dallas." in stripped
    assert "relevant memories" not in stripped.lower()
    assert "mempalace-recall" not in stripped.lower()


def test_strip_injected_leaves_ordinary_prose_alone():
    text = "I checked my memory for the tool manifest and found nothing about the dashboard."

    assert strip_injected(text) == text


# === classify: exact-literal, and conservative in the one direction that matters ===============


def test_a_lone_legacy_heading_is_scrubbable():
    verdict, classes = classify(LEGACY_HEADING)

    assert verdict is Verdict.SCRUB
    assert classes == ("recall-block",)


def test_bulleted_and_quoted_copies_still_read_as_scaffolding():
    """A mined chunk carries MemPalace's ``>`` quote markers and the recall renderer's bullets."""
    chunk = f"> - {LEGACY_HEADING}\n- {LEGACY_HEADING}\n>\n- \n"

    assert classify(chunk)[0] is Verdict.SCRUB


def test_scaffolding_beside_real_content_is_review_and_never_scrub():
    """The safety property: a chunk holding a memory is never deleted, whatever else is in it."""
    chunk = f"{LEGACY_HEADING}\n- John's birthday is 3 March."

    verdict, classes = classify(chunk)

    assert verdict is Verdict.REVIEW
    assert classes == ("recall-block",)


def test_a_memory_that_merely_mentions_memory_and_tools_is_kept():
    chunk = (
        "> [2026-08-29] john: what do you remember about the tool manifest in your brief?\n"
        "I recall we discussed the dashboard and the memory search tool last March."
    )

    assert classify(chunk) == (Verdict.KEEP, ())


def test_a_short_line_is_never_enough_to_prove_scaffolding():
    """`MIN_UNIT_CHARS` is the anti-promiscuity rule: a fragment proves nothing."""
    assert len("Your active tools right now:") < MIN_UNIT_CHARS
    assert classify("Your active tools right now:")[0] is Verdict.KEEP


def test_every_catalog_literal_classifies_as_scaffolding():
    """Self-consistency: a class the catalog names must be one the classifier can act on.

    A literal shorter than `MIN_UNIT_CHARS` is deliberately unmatchable, so those are exempt —
    they exist in the catalog as documentation of a class, and are matched via their longer
    siblings or their prefix.
    """
    for entry in catalog():
        for literal in entry.literals:
            if len(" ".join(literal.split())) < MIN_UNIT_CHARS:
                continue
            verdict, classes = classify(literal)
            assert verdict is Verdict.SCRUB, f"{entry.name}: {literal[:60]!r}"
            assert entry.name in classes


def test_the_catalog_carries_this_boxs_own_charter(monkeypatch, tmp_path):
    """An operator's charter is what actually reached the brief, so it is what the catalog names.

    The packaged default is only the fallback: on a real agent `system-prompt.md` is edited, and a
    catalog holding the shipped text alone could not recognize a mined copy of the real one.
    """
    home = tmp_path / "config-home"
    install(home)  # a real reconcile, so the config home is authoritative for prompts
    charter = (
        "You are Nova Digital, and you keep every promise you make to a peer on this platform.\n"
        "\n"
        "## Voice\n"
        "\n"
        "Plain, direct, never florid. Say the thing and stop talking."
    )
    (home / "prompts" / "system-prompt.md").write_text(charter, encoding="utf-8")
    monkeypatch.setenv("BASECRADLE_CONFIG_HOME", str(home))
    _mining._reset()

    assert classify(charter)[0] is Verdict.SCRUB
    # And it is the *charter* class saying so, not an accident of another literal.
    assert "charter" in classify(charter)[1]


def test_the_catalog_is_read_from_the_constants_the_harness_writes():
    """Never a re-typed copy: the catalog moves when the wording does (see `_mining`)."""
    literals = {literal for entry in catalog() for literal in entry.literals}

    assert wake_module._STUCK_NOTE in literals
    assert wake_module._COMPACTION_OBSERVE_NOTE in literals
    assert wake_module._NOW_LINE_INSTRUCTION in literals
    assert _INJECTED_HEADING in literals


def test_scope_is_the_agent_not_the_timeline(platform, tmp_path):
    """Unchanged by this issue, and re-pinned here: memory is one mind across every channel."""
    provider = Recorder(injected=None)
    agent = _agent(tmp_path, provider)

    agent.wake()

    assert provider.observed[-1].scope == MemoryScope(agent=agent.me_uuid, timeline=TIMELINE_UUID)
