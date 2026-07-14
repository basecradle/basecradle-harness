"""A posted asset, an inbound webhook delivery, and an activated task are not dropped either.

Issue #289. Issue #285 closed the silent permanent drop on the **message** path: a wake claimed an
item and marked it seen *before* it called the model, so any hard failure in between left the item
recorded, never acted on, and never looked at again by any future wake. `_act_on` — the one loop
behind the other three reconcilers — had the identical shape, and so the identical drop. Messages
were fixed first because a peer silently ignored is the highest harm, **not** because the other
three were safe.

The finding, and it is the reason this is a small change rather than a second mechanism: **the
recovery is kind-agnostic.** The issue anticipated a per-kind *post-landed test* — did the asset's
reply reach the timeline? did the webhook's? — because when it was filed, the message path had one:
the harness held a reply it still had to deliver, and recovery reconciled it against the timeline by
body-equality. The Unspoken Channel (#293) deleted that question rather than answering it. The
harness holds no reply, for any kind: everything an agent says, it said itself, mid-turn, through
the `messages` tool. So the only question left is *did the turn finish?*, and the transcript answers
it identically for all four kinds.

What genuinely differs is the **queue** an unsettled item comes back on, and these tests pin both:

- A **mark-backed** kind (assets, webhook deliveries) rides a cursor, so the mark must stop dead at
  an unsettled item — passing it hides it forever.
- A **seen-backed** kind (tasks) has no cursor to hold back, and needs none: the queue is the
  platform's own `activated` list, so a task stays on it until this agent records it. An undecided
  task holds back only itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

import httpx
import pytest
import respx

from basecradle_harness import Harness, Message, MessagesTool, Session, ToolCall
from basecradle_harness._exceptions import ProviderContextLengthError
from basecradle_harness._idempotency import MESSAGE, key
from basecradle_harness._wake import ClaimStore, MarkStore, SeenStore, WakeAgent
from tests.test_wake import (
    A0,
    A1,
    BC_URL,
    E0,
    E1,
    PNG_BYTES,
    REPLY,
    T0,
    T1,
    TIMELINE_UUID,
    CountingProvider,
    _posts,
    asset,
    asset_page,
    build_wake,
    crashed_wake_owning,
    dashboard,
    event,
    event_page,
    live_wake_owning,
    message,
    page,
    serve_messages,
    task,
    task_page,
    timeline,
)


@pytest.fixture
def platform():
    """The respx-mocked platform — its own copy, so the fixture name is not an imported symbol."""
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
        router.get(path__regex=r"^/blobs/").mock(
            return_value=httpx.Response(200, content=PNG_BYTES)
        )
        yield router


@pytest.fixture
def kill_the_finally(monkeypatch):
    """Simulate a signal: the incremental writes land, the closing one never happens.

    `Session._persist` is the save in `_drive`'s `finally`. A `finally` runs on an exception and
    **not** on a signal, so neutering exactly this one — and nothing else — is what a `SIGKILL`
    looks like from the disk's point of view (the same fixture `test_resume` uses, for the same
    reason).
    """
    monkeypatch.setattr(Session, "_persist", lambda self, *, masking: None)


# --- the three reconcilers, as one table -------------------------------------
#
# The guarantee is the same for all three, so the tests are written once and parametrized over
# them. Where a kind genuinely differs — how its record refuses to pass an unsettled item — it gets
# its own test below, and the difference is the *point* of that test rather than a branch in a
# shared one.


@dataclass(frozen=True)
class Reconciler:
    """One of the three kinds `_act_on` drives, and everything a test needs to exercise it."""

    kind: str
    prior: str  # an item already handled before the test begins
    item: str  # the item under test — unseen, and the one a dead wake will strand
    probe: str  # a NOC synthetic probe, older than `item` (only the probe-ack test uses it)
    path: str  # the SDK read surface it is scanned from
    wire: Callable[[str], dict]  # uuid -> the item's wire payload
    probe_wire: Callable[
        [str], dict
    ]  # uuid -> the same, with "PROBE" in this kind's marker carrier
    envelope: Callable[..., dict]  # (*items) -> the newest-first page
    handled: Callable[[object, str], bool]  # (home, uuid) -> has this agent recorded it?
    record: Callable[[object, str], None]  # (home, uuid) -> record it as handled
    rewind: Callable[[object], None]  # (home) -> put the record back to "only `prior` is handled"
    reads: str  # a fragment of the rendered text, so a test can prove the model was shown it


def _asset_handled(home, uuid) -> bool:
    return MarkStore(home).get(TIMELINE_UUID, kind="assets") == uuid


def _event_handled(home, uuid) -> bool:
    return MarkStore(home).get(TIMELINE_UUID, kind="webhook_events") == uuid


def _task_handled(home, uuid) -> bool:
    return uuid in SeenStore(home).all(TIMELINE_UUID, kind="tasks")


def _task_rewind(home) -> None:
    """Un-record every task but `prior` — a seen-set is a **set**, so rewinding it means removal.

    Rewinding a *mark* is a write (point the cursor back); rewinding a set is a *deletion*, and
    conflating the two is a test that pins nothing: adding the older task to the set leaves the
    newer one in it, so the scan filters the newer one out and the recovery under test never runs.
    """
    SeenStore(home)._path(TIMELINE_UUID, "tasks").write_text(f"{T0}\n")


# A probe sits *between* `prior` and `item` in each kind's stream, so the probe-ack test can put an
# undecided item behind a settled one. Well-formed UUIDv7s, ordered accordingly.
A_OLD = "019e777f-0000-7aaa-8bbb-617283940516"  # older than A0: a mark to scan from
A_PROBE = "019e7780-eeee-7fff-8aaa-051627384950"
E_PROBE = "019e7761-eeee-7fff-8aaa-162738495061"
T_PROBE = "019e7770-eeee-7fff-8aaa-273849506172"

ASSETS = Reconciler(
    kind="assets",
    prior=A0,
    item=A1,
    probe=A_PROBE,
    path="/assets",
    wire=lambda uuid: asset(uuid=uuid, filename="owl.png"),
    # An asset's probe marker is read from its **description** (`_asset_marker_carrier`).
    probe_wire=lambda uuid: asset(uuid=uuid, filename="probe.png", description="PROBE"),
    envelope=asset_page,
    handled=_asset_handled,
    record=lambda home, uuid: MarkStore(home).set(TIMELINE_UUID, uuid, kind="assets"),
    rewind=lambda home: MarkStore(home).set(TIMELINE_UUID, A0, kind="assets"),
    reads="owl.png",
)

EVENTS = Reconciler(
    kind="webhook_events",
    prior=E0,
    item=E1,
    probe=E_PROBE,
    path="/webhook_events",
    wire=lambda uuid: event(uuid=uuid, payload='{"deploy":"finished"}'),
    # A delivery's probe marker is read from its **payload**.
    probe_wire=lambda uuid: event(uuid=uuid, payload="PROBE"),
    envelope=event_page,
    handled=_event_handled,
    record=lambda home, uuid: MarkStore(home).set(TIMELINE_UUID, uuid, kind="webhook_events"),
    rewind=lambda home: MarkStore(home).set(TIMELINE_UUID, E0, kind="webhook_events"),
    reads="deploy",
)

TASKS = Reconciler(
    kind="tasks",
    prior=T0,
    item=T1,
    probe=T_PROBE,
    path="/tasks",
    wire=lambda uuid: task(uuid=uuid, instructions="water the plants"),
    # A task's probe marker is read from its **instructions**.
    probe_wire=lambda uuid: task(uuid=uuid, instructions="PROBE"),
    envelope=task_page,
    handled=_task_handled,
    record=lambda home, uuid: SeenStore(home).add(TIMELINE_UUID, uuid, kind="tasks"),
    rewind=_task_rewind,
    reads="water the plants",
)

KINDS = [pytest.param(r, id=r.kind) for r in (ASSETS, EVENTS, TASKS)]


def serve(platform, reconciler: Reconciler, *uuids):
    """Serve `uuids` (newest-first) on this kind's read surface, for every read of this wake or any
    later one — the platform does not forget an item just because a wake died holding it."""

    def wire(uuid):
        return (reconciler.probe_wire if uuid == reconciler.probe else reconciler.wire)(uuid)

    body = reconciler.envelope(*(wire(uuid) for uuid in uuids))
    platform.get(reconciler.path).mock(return_value=httpx.Response(200, json=body))


def stage(platform, tmp_path, reconciler: Reconciler):
    """The timeline as it stands before each test: `prior` handled, `item` unseen and waiting."""
    serve_messages(platform, page())  # the message path is a clean no-op throughout
    serve(platform, reconciler, reconciler.item, reconciler.prior)
    reconciler.record(tmp_path, reconciler.prior)


def shown(brain, uuid: str) -> bool:
    """Was **this** item put in front of the model? Asserted by uuid, never by a text fragment.

    Every renderer embeds the item's uuid (`_describe`, `_incoming_event_text`,
    `_activated_task_text`), so the uuid is free — and a fragment of the *kind's* boilerplate would
    pass just as happily if the wake had rendered some other item of the same kind twice.
    """
    return any(uuid in prompt for prompt in brain.prompts)


def settled(agent, reconciler: Reconciler, uuid: str) -> bool:
    """Did the claim actually reach a final phase? The record moving is only half the commit.

    Worth asserting separately: a `_settle` that advanced the mark but never committed the claim
    would leave the item looking orphaned to the next wake forever, and every assertion about the
    *record* would still pass.
    """
    claim = agent.claims.read(TIMELINE_UUID, uuid, kind=reconciler.kind)
    return claim is not None and claim.settled


class DiesInTheModelCall:
    """The provider is down, the box is killed — the model is called and never answers."""

    provider, model = "openai", "gpt-4o"
    prompts: list[str] = []

    def chat(self, messages, tools=None):
        raise RuntimeError("the model call died")


class Speaks:
    """Posts one message through the `messages` tool, then dies — the killed-after-posting wake."""

    provider, model = "openai", "gpt-4o"

    def __init__(self, body: str = "On it.") -> None:
        self.body = body
        self.calls = 0

    def chat(self, messages, tools=None):
        self.calls += 1
        if self.calls == 1:
            return Message.assistant(
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="messages",
                        arguments={"action": "create", "body": self.body},
                    )
                ]
            )
        raise RuntimeError("SIGKILL: the box went down after the message was posted")


class Finishes:
    """Picks an interrupted turn up and settles it, saying nothing new."""

    provider, model = "openai", "gpt-4o"

    def __init__(self) -> None:
        self.seen: list[list[Message]] = []
        self.prompts: list[str] = []

    def chat(self, messages, tools=None):
        self.seen.append(list(messages))
        return Message.assistant(content="Already handled that; nothing more to add.")


# --- the headline: the record no longer moves before the work is done ---------


@pytest.mark.parametrize("kind", KINDS)
def test_a_wake_that_dies_leaves_the_item_for_the_next_wake(platform, tmp_path, kind):
    """**The bug, closed, for all three kinds.** A dead wake's item is still there afterwards.

    This is issue #289 in one test. `_act_on` used to `record(item)` — advance the mark, add to the
    seen-set — *before* engaging the model, so the wake below (whose model call dies) would have
    left the item filed as handled and invisible to every later wake: a file nobody looked at, a
    delivery nobody read, a task nobody ran, all silently marked done.

    Now the record moves only at `_settle`, over settled items. The item is still unhandled when the
    dead wake is gone, so the next wake finds it, acts on it, and *then* records it.
    """
    stage(platform, tmp_path, kind)

    first, _ = build_wake(tmp_path, DiesInTheModelCall())
    with pytest.raises(RuntimeError):
        first.wake()
    assert not kind.handled(tmp_path, kind.item)  # NOT recorded: the work never happened

    second, brain = build_wake(tmp_path)
    second.wake()
    assert shown(brain, kind.item)  # the model was shown *this* item
    assert kind.handled(tmp_path, kind.item)  # and only now is it recorded
    assert settled(second, kind, kind.item)


@pytest.mark.parametrize("kind", KINDS)
def test_a_dead_wake_that_never_reached_the_model_is_re_driven(platform, tmp_path, kind, caplog):
    """Outcome 1: a claim, and no turn carrying it. The wake died before the model ever saw it."""
    stage(platform, tmp_path, kind)
    crashed_wake_owning(tmp_path, kind.item, kind=kind.kind)

    agent, brain = build_wake(tmp_path)
    with caplog.at_level("WARNING", logger="basecradle_harness"):
        agent.wake()

    assert shown(brain, kind.item)  # re-driven: the model sees it
    assert kind.handled(tmp_path, kind.item)
    assert settled(agent, kind, kind.item)
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("re-driving"))
    assert f"item={kind.item}" in line and f"kind={kind.kind}" in line


@pytest.mark.parametrize("kind", KINDS)
def test_a_dead_wake_whose_turn_finished_is_committed_never_re_run(platform, tmp_path, kind):
    """Outcome 2: the turn reached its terminal narration, so the model **finished**.

    The reachable shape of this is a wake killed in the window between persisting its narration and
    settling its claim — which is staged here by running a real wake and then rewinding only the
    bookkeeping it would not have got to. The transcript is genuine; the claim is orphaned.
    """
    stage(platform, tmp_path, kind)
    first, _ = build_wake(tmp_path)
    first.wake()
    assert _posts(platform) == ["Hello, John."]

    kind.rewind(tmp_path)  # rewind the record: the dead wake never advanced it
    crashed_wake_owning(tmp_path, kind.item, kind=kind.kind)  # ...nor settled its claim

    second, brain = build_wake(tmp_path, Finishes())
    second.wake()
    assert brain.seen == []  # the model is never consulted: that turn is done
    assert _posts(platform) == ["Hello, John."]  # and nothing is said a second time
    assert kind.handled(tmp_path, kind.item)  # the record catches up
    assert settled(second, kind, kind.item)


@pytest.mark.parametrize("kind", KINDS)
def test_a_dead_wake_that_already_spoke_is_resumed_never_re_driven(
    platform, tmp_path, kind, kill_the_finally
):
    """Outcome 3 — **the one that matters.** A turn whose tools fired is finished, never re-run.

    A wake posts through the `messages` tool and is killed by a signal. Its results are on disk, so
    the turn needs neither re-running (which would post again) nor abandoning (which would drop the
    peer): it needs *continuing*.

    **The post count alone would be a trap, and it is worth saying why**, because it is the obvious
    assertion and it cannot fail: under the *old* code the dying wake recorded the item before it
    engaged, so the second wake's scan never surfaced it, nothing was recovered, and exactly one post
    stands — the right number, for the worst possible reason. So the resume must be asserted
    positively: the model **was** engaged (`brain.seen`), it was handed the *interrupted turn* rather
    than a fresh copy of the item, and only then does the post count mean what it looks like it means.
    """
    stage(platform, tmp_path, kind)

    first, _ = build_wake(tmp_path, Speaks())
    with pytest.raises(RuntimeError):
        first.wake()
    assert _posts(platform) == ["On it."]  # the dead wake did speak

    second, brain = build_wake(tmp_path, Finishes(), tools=[MessagesTool()])
    second.wake()

    assert brain.seen, "the turn was never resumed — the item was silently dropped, not recovered"
    replayed = brain.seen[0]
    # It was handed the turn it was cut off in: the item, its own tool call, and that call's result.
    assert any(m.tool_calls and m.tool_calls[0].name == "messages" for m in replayed)
    assert sum(1 for m in replayed if m.role == "user" and not m.injected) == 1
    assert _posts(platform) == ["On it."]  # ONE post, not two: the tool never re-fired
    assert kind.handled(tmp_path, kind.item)
    assert settled(second, kind, kind.item)


# --- what actually differs per kind: how the record refuses to pass an item ---


@pytest.mark.parametrize("kind", [pytest.param(r, id=r.kind) for r in (ASSETS, EVENTS)])
def test_the_mark_never_passes_an_item_a_live_wake_still_holds(platform, tmp_path, kind):
    """A mark is a **cursor**, so it must stop dead at an undecided item — passing it hides it.

    A concurrent wake holds the older item in flight. This wake skips it (correctly — it is owned)
    and acts on the newer one, but the mark may not advance past the item it skipped: if that wake
    then died, a mark beyond it would hide it from every future wake. So the mark stays put, and
    both items are re-scanned next wake — a timeline read, never a model call.
    """
    serve_messages(platform, page())
    serve(platform, kind, kind.item, kind.prior)
    live_wake_owning(tmp_path, kind.prior, kind=kind.kind)  # in flight, owner still running

    agent, brain = build_wake(tmp_path)
    agent.wake()

    assert len(brain.prompts) == 1  # the newer item was acted on...
    assert not kind.handled(tmp_path, kind.item)  # ...but the mark did not sail past the older one
    assert MarkStore(tmp_path).get(TIMELINE_UUID, kind=kind.kind) is None


def test_a_task_a_live_wake_holds_does_not_suppress_the_record_of_the_others(platform, tmp_path):
    """A seen-set is a **set**, so an undecided task holds back only itself — never the queue.

    This is the explicit answer to "what replaces *the un-advanced mark is the queue*" for tasks.
    Nothing does, because nothing needs to: the queue is the **platform's** `activated` list, and a
    task stays on it until this agent records it. So withholding one task's seen-set entry re-offers
    exactly that task next wake — while every task around it settles normally.

    Applying the cursor rule here would be strictly worse than useless: one task a concurrent wake
    happened to hold would suppress the record of every task behind it, and those would then be
    re-driven on the next wake — turns re-run, tools re-fired — for nothing.
    """
    serve_messages(platform, page())
    serve(platform, TASKS, T1, T0)  # both activated, neither seen
    live_wake_owning(tmp_path, T0, kind="tasks")  # a concurrent wake is mid-turn on T0

    agent, brain = build_wake(tmp_path)
    agent.wake()

    assert len(brain.prompts) == 1  # T1 was acted on
    seen = SeenStore(tmp_path).all(TIMELINE_UUID, kind="tasks")
    assert T1 in seen  # ...and recorded, on its own
    assert T0 not in seen  # while T0 stays on the queue for whoever finishes it


@pytest.mark.parametrize("kind", [pytest.param(r, id=r.kind) for r in (ASSETS, EVENTS)])
def test_a_first_wake_does_not_baseline_over_an_orphan(platform, tmp_path, kind):
    """The bootstrap's baseline is a **jump**, not a step — so it is where the guard can be leapt.

    Now that the mark advances at settle rather than at claim, a wake that dies on the very first
    item of a timeline leaves **no mark at all**. The next wake therefore bootstraps — and a
    bootstrap acts on the newest item only. If anything landed in the meantime, the orphan is older
    than the newest, so it would be stepped over *and baselined past*: gone forever, which is the
    drop this issue exists to remove, sneaking back in through the one path that jumps the cursor.

    `_extend_over_unfinished` widens the act-set back over it. Both items are acted on.
    """
    serve_messages(platform, page())
    serve(platform, kind, kind.item, kind.prior)  # no mark: this is a cold first wake
    crashed_wake_owning(tmp_path, kind.prior, kind=kind.kind)  # a dead wake claimed the older one

    agent, brain = build_wake(tmp_path)
    agent.wake()

    assert shown(brain, kind.prior), "the orphan was baselined past — the drop, straight back in"
    assert shown(brain, kind.item)  # ...and the newest, which is what the bootstrap came for
    assert kind.handled(tmp_path, kind.item)
    assert settled(agent, kind, kind.prior)


@pytest.mark.parametrize("kind", [pytest.param(r, id=r.kind) for r in (ASSETS, EVENTS)])
def test_a_first_wake_recovers_an_orphan_its_read_window_cannot_see(platform, tmp_path, kind):
    """A bounded window cannot protect an orphan a **burst** pushed out of it.

    `_extend_over_unfinished` widens the act-set back over an orphan *inside* the window — and that
    is all it can do. The window is 50 items. A wake claims a delivery and dies; sixty more land
    while the agent is down; the orphan is now the 61st-newest and the bootstrap cannot see it at
    all. It would act on the newest, baseline the mark past everything, and a cursor never looks
    back: gone, silently and permanently.

    So the unfinished work is read from the **claims**, which know what the window does not, and the
    orphan is fetched by uuid. `context_messages=1` here is the same geometry as 50-vs-61, without
    building sixty items to say it.
    """
    serve_messages(platform, page())
    serve(platform, kind, kind.item, kind.prior)
    crashed_wake_owning(tmp_path, kind.prior, kind=kind.kind)
    # The one item the window can hold is the *newest* — so the orphan is outside it entirely.
    platform.get(f"{kind.path}/{kind.prior}").mock(
        return_value=httpx.Response(200, json={kind.kind.rstrip("s"): kind.wire(kind.prior)})
    )

    agent, brain = build_wake(tmp_path, context_messages=1)
    agent.wake()

    assert shown(brain, kind.prior), "an orphan outside the read window was silently baselined past"
    assert settled(agent, kind, kind.prior)


# --- a compaction may destroy a turn, so a missing turn must be safe ----------


@pytest.mark.parametrize("kind", KINDS)
def test_a_turn_a_compaction_destroyed_is_abandoned_not_re_driven(platform, tmp_path, kind, caplog):
    """**The classifier's one inference, and the one thing that falsifies it.**

    *No turn carries this item ⟹ the model never saw it ⟹ nothing ran ⟹ re-drive.* Compaction makes
    that a lie: it replaces a region of the transcript with a single summary, so a turn that ran
    tools — that posted, that bought an image at fal.ai — ceases to exist while the item's claim is
    still in flight. Re-driving it re-fires everything it fired.

    And it is worse than a plain double-post, which is why abandoning is the right trade. The keys
    are deterministic, so the re-driven turn mints the key the *dead* turn used: the platform hands
    back the **original** record, the tool reports success, the model believes it spoke — and the
    reply it actually composed reaches nobody. A duplicate wearing a drop's clothes.

    So a missing turn whose uuid a summary is carrying is abandoned, loudly. (A missing turn whose
    uuid is *nowhere* was genuinely never sent, and still re-drives — the test above.)
    """
    stage(platform, tmp_path, kind)
    crashed_wake_owning(tmp_path, kind.item, kind=kind.kind)

    # The transcript a compaction left behind: the turn is gone, and the summary says it was here.
    dead = Harness(Finishes(), home=tmp_path).session(f"timeline:{TIMELINE_UUID}")
    erased = Message.system("[Earlier conversation summarized] ... it did something ...")
    erased.items = [kind.item]
    dead.history.append(erased)
    dead._save()

    agent, brain = build_wake(tmp_path, Finishes())
    with caplog.at_level("ERROR", logger="basecradle_harness"):
        agent.wake()

    assert brain.seen == [], "a turn whose evidence a compaction destroyed was re-driven"
    assert _posts(platform) == []  # nothing said again, nothing swallowed
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("dropped"))
    assert f"item={kind.item}" in line and f"kind={kind.kind}" in line
    # Abandoned is *settled*, so the record may pass it: a loud drop, never a permanent stall.
    assert kind.handled(tmp_path, kind.item)


def test_the_evidence_a_compaction_carries_is_released_once_the_item_settles(platform, tmp_path):
    """**What bounds the uuids a summary carries** — the Context Discipline half of the fix.

    A summary inherits the uuids of the turns it destroyed so the recovery can tell "never seen" from
    "seen, evidence gone". Left alone that list would grow by one uuid per item ever handled on the
    timeline, forever — the unbounded-content defect, even though these uuids cost no tokens.

    A uuid is evidence only while its item's disposition is *undecided*: the recovery reads it only
    for a claim still in flight. So once the claim is final the uuid has no reader left, and the wake
    releases it. Here A0 is long settled and A1 is still in flight — only A1 survives the prune.
    """
    serve_messages(platform, page())
    serve(platform, ASSETS, A0)
    MarkStore(tmp_path).set(TIMELINE_UUID, A0, kind="assets")
    ClaimStore(tmp_path).claim(TIMELINE_UUID, A0, kind="assets")
    ClaimStore(tmp_path).commit(TIMELINE_UUID, A0, kind="assets")  # settled: nobody will ask again
    live_wake_owning(tmp_path, A1, kind="assets")  # still in flight: its evidence must be kept

    session = Harness(Finishes(), home=tmp_path).session(f"timeline:{TIMELINE_UUID}")
    summary = Message.system("[Earlier conversation summarized] ...")
    summary.items = [
        A0,
        A1,
        "019e7799-0000-7000-8000-000000000000",
    ]  # settled, in-flight, unclaimed
    session.history.append(summary)
    session._save()

    agent, _ = build_wake(tmp_path, Finishes())
    agent.wake()

    kept = next(m for m in agent.harness.session(agent.source).history if m.role == "system").items
    assert kept == [A1], "the summary is hoarding evidence nobody will ever come looking for"


def test_a_finished_items_claim_settles_before_the_next_item_runs(platform, tmp_path):
    """A claim settles the instant its turn ends — **not** at the end of the reconcile.

    `_act_on` runs one turn *per item*, and a later item's turn can compact the transcript (an
    over-length rescue compacts hard, and its keep is a fraction of what is actually there — so this
    needs no enormous transcript). A compaction is destructive: it can summarize away an *earlier*
    item's finished turn. If that item's claim were still in flight when it happened, the next wake
    would find an orphan whose turn is gone.

    Committing at the end of the loop would leave exactly that window open, for as long as the items
    behind it take. Committing when the turn ends closes it: a dead wake leaves exactly **one**
    in-flight claim per kind — the item it died on — and everything it finished is beyond re-drive.
    """
    serve_messages(platform, page())
    # Two unseen assets past the mark, acted on oldest-first — so the wake dies *behind* A0.
    serve(platform, ASSETS, A1, A0, A_OLD)
    MarkStore(tmp_path).set(TIMELINE_UUID, A_OLD, kind="assets")

    class DiesOnTheSecond:
        provider, model = "openai", "gpt-4o"
        prompts: list[str] = []

        def __init__(self):
            self.calls = 0

        def chat(self, messages, tools=None):
            self.calls += 1
            if self.calls == 1:
                return Message.assistant(content="Looked at the first one.")
            raise RuntimeError("the box went down on the second asset")

    agent, _ = build_wake(tmp_path, DiesOnTheSecond())
    with pytest.raises(RuntimeError):
        agent.wake()

    # `_settle` never ran — the wake died inside the second item's turn. The first item's claim is
    # settled anyway, because its *turn* ended, which is the only thing that decides it.
    first = agent.claims.read(TIMELINE_UUID, A0, kind="assets")
    assert first is not None and first.phase == "done"
    second = agent.claims.read(TIMELINE_UUID, A1, kind="assets")
    assert second is not None and second.phase == "in-flight"  # the one it died on, and only that


# --- a drop is never silent ---------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_an_abandoned_item_is_an_error_naming_the_item_and_its_kind(
    platform, tmp_path, kind, caplog
):
    """The residual at-most-once drop stays **loud**, and now says which kind of item it lost.

    Reached only when a dead wake's interrupted turn cannot be resumed because it has outgrown the
    model's context window and cannot be compacted — retrying would fail identically, forever, with
    the record pinned behind it. Rare, bounded, and an ERROR naming the item: a drop that nobody can
    see is the failure class this whole issue is about.
    """
    stage(platform, tmp_path, kind)

    first, _ = build_wake(tmp_path, Speaks())
    with pytest.raises(RuntimeError):
        first.wake()  # leaves an interrupted turn: a tool call, no narration

    class OverTheCeiling:
        provider, model = "openai", "gpt-4o"

        def chat(self, messages, tools=None):
            raise ProviderContextLengthError(
                "maximum context length exceeded", status_code=400, body=""
            )

    second, _ = build_wake(tmp_path, OverTheCeiling())
    with caplog.at_level("ERROR", logger="basecradle_harness"):
        second.wake()

    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("dropped"))
    assert f"item={kind.item}" in line and f"kind={kind.kind}" in line
    # Abandoned is *settled*, so the record may pass it: a loud drop, never a permanent stall.
    assert kind.handled(tmp_path, kind.item)


# --- the keys were minted for these kinds before the recovery existed ---------


def test_an_interrupted_create_in_a_task_turn_is_re_issued_under_the_dead_wakes_key(
    platform, tmp_path
):
    """The idempotency key an activated task's turn mints is the one its recovery replays.

    `_engage` has anchored every asset/webhook/task turn's keys since #297, on the bet that the day
    their recovery landed the keys would already be right. This is that day, and this is the proof:
    the resumed wake re-issues the dead wake's interrupted `messages` create under a key derived
    from the **task's** uuid — so the platform returns the original record instead of posting twice.

    The transcript is the one genuinely unknowable state, so it is staged rather than driven: the
    dead wake issued the create and was killed before its result could be written, and the POST may
    or may not have landed. The key is what makes acting on that safely possible at all.
    """
    stage(platform, tmp_path, TASKS)
    crashed_wake_owning(tmp_path, T1, kind="tasks")

    # The dead wake's transcript, written by a *different* Harness — so the wake below loads it from
    # disk and heals the dangling call, exactly as it would after a real kill. Mutating the live
    # session instead would skip `_load`, and with it the healing the recovery reads.
    dead = Harness(Finishes(), home=tmp_path).session(f"timeline:{TIMELINE_UUID}")
    dead.history.append(Message(role="user", content="[task] water the plants", items=[T1]))
    dead.history.append(
        Message.assistant(
            tool_calls=[
                ToolCall(
                    id="c1", name="messages", arguments={"action": "create", "body": "Watered."}
                )
            ]
        )
    )
    dead._save()

    agent, _ = build_wake(tmp_path, Finishes(), tools=[MessagesTool()])
    agent.wake()

    posts = [
        call.request
        for call in platform.calls
        if call.request.method == "POST" and call.request.url.path.endswith("/messages")
    ]
    # The re-issue carries the key derived from the **task** the turn was answering — not a fresh
    # one, and not a message's. A key the platform has never seen is a message posted twice.
    assert [p.headers.get("idempotency-key") for p in posts] == [
        key(timeline=TIMELINE_UUID, anchor=T1, kind=MESSAGE, ordinal=1)
    ]
    assert json.loads(posts[0].content)["message"]["body"] == "Watered."  # replayed byte for byte


# --- the self-filter and the probe seam still short-circuit before the claim --


def test_the_agents_own_asset_is_recorded_without_ever_being_claimed(platform, tmp_path):
    """The actor self-filter: an image the agent generated is final at once, and never acted on.

    It needs no claim (there is nothing to make exclusive) and it must still advance the mark, or
    the agent re-scans its own output forever. `_settle` treats it as settled, which is what keeps
    the wake-loop closed while the record still moves.
    """
    serve_messages(platform, page())
    body = asset_page(asset(uuid=A1, filename="mine.png", mine=True), asset(uuid=A0))
    platform.get("/assets").mock(return_value=httpx.Response(200, json=body))
    MarkStore(tmp_path).set(TIMELINE_UUID, A0, kind="assets")

    agent, brain = build_wake(tmp_path)
    agent.wake()

    assert brain.prompts == []  # never acted on
    assert _asset_handled(tmp_path, A1)  # but recorded, so it is not re-scanned forever
    assert agent.claims.read(TIMELINE_UUID, A1, kind="assets") is None  # and never claimed


@pytest.mark.parametrize("kind", [pytest.param(r, id=r.kind) for r in (ASSETS, EVENTS)])
def test_a_probe_whose_ack_failed_is_not_marked_seen_by_the_item_behind_it(
    platform, tmp_path, kind, monkeypatch
):
    """The same cursor-versus-set confusion, in miniature — and it was a live false-FAIL.

    A NOC probe's ack is at-least-once: a refused post leaves the item deliberately unrecorded, so
    the next wake re-acks it. But `_act_on` recorded each item *as it went*, and for a mark-backed
    kind the record is a **cursor** — so the very next item in the batch advanced the mark straight
    past the un-acked probe, and nobody ever acked it: a false monitor FAIL, manufactured by the
    code written to prevent one. (The reasoning was right for tasks, whose record really is a set,
    and quietly wrong for the other two.)

    The order is the whole test: the probe is **older** than the real item, so a per-item record
    sails over it. The mark now stops dead at the first undecided item, so it stays where it was and
    the probe is re-acked next wake.
    """
    serve_messages(platform, page())
    serve(platform, kind, kind.item, kind.probe, kind.prior)  # newest-first: real, probe, handled
    kind.record(tmp_path, kind.prior)  # the incremental path, so all three are in one batch
    monkeypatch.setattr(
        WakeAgent,
        "_probe_nonce",
        lambda self, carrier: "n-1234" if "PROBE" in str(carrier) else None,
    )
    monkeypatch.setattr(WakeAgent, "_post", lambda self, body, *, kind: None)  # the ack is refused

    agent, brain = build_wake(tmp_path)
    agent.wake()

    assert len(brain.prompts) == 1  # the real item was still acted on
    # ...but the mark did not sail past the un-acked probe sitting behind it. It stays put, and the
    # next wake re-reads both — a timeline read, never a model call.
    assert MarkStore(tmp_path).get(TIMELINE_UUID, kind=kind.kind) == kind.prior
    assert not kind.handled(tmp_path, kind.item)


def test_a_working_wake_records_and_commits_every_kind_it_handled(platform, tmp_path):
    """The happy path, end to end: one wake, all three kinds, each settled and each recorded."""
    serve_messages(platform, page())
    serve(platform, ASSETS, A0)
    serve(platform, EVENTS, E0)
    serve(platform, TASKS, T0)

    agent, brain = build_wake(tmp_path, CountingProvider())
    agent.wake()

    assert len(brain.prompts) == 3  # one turn per item
    for kind, uuid in ((ASSETS, A0), (EVENTS, E0), (TASKS, T0)):
        assert kind.handled(tmp_path, uuid)
        assert agent.claims.read(TIMELINE_UUID, uuid, kind=kind.kind).phase == "done"
