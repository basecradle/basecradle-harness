"""The standalone `LockTool`, behind the shared uuid-confirm gate, against a respx platform.

Lock is one of two irreversible timeline actions (delete is the other); both run through the
one `ConfirmedTimelineAction` gate (see `_confirmed.py` and `test_delete.py`). Group 2b pulled
lock out of the `timelines` tool into its own tool so a benign management call can never grab
it by accident (finding B1); this re-unifies its gate with delete's: it acts only when
`confirm` equals the **target timeline's uuid**, and a bare or mismatched call previews what
would be frozen and touches nothing destructive. The headline invariants: a non-matching
confirm never locks (only a benign preview GET goes out), and `confirm=<uuid>` actually locks —
the current timeline by default, or an explicit one.

Mocked at the transport level (a real `BaseCradle` client, only its HTTP stubbed), per the
constitution. Cast: Nova Digital (``nova``, AI) freezing her own room.
"""

import httpx
import pytest
import respx
from basecradle import BaseCradle

from basecradle_harness import LockTool, PlatformContext, PlatformError

BC_URL = "https://basecradle.com"
FAKE_TOKEN = "bc_uat_KqI8zFxkQ0OZ8vYwT7mWcVtR3nSdLpEa"

NOVA_UUID = "019e7750-66ee-79c8-ad8a-bbb6ea7c2bcc"
TIMELINE_UUID = "019e7750-66ee-7f53-829f-13a8a710b6da"  # the current timeline
OTHER_TIMELINE = "019e7760-1234-7abc-8def-0123456789ab"  # an explicit override


def timeline_envelope(*, uuid=TIMELINE_UUID, name="Incident response", locked=False, items=0):
    """The subject-timeline envelope the SDK merges on get."""
    return {
        "timeline": {
            "uuid": uuid,
            "name": name,
            "locked": locked,
            "created_at": "2026-06-01T00:00:00.000Z",
            "updated_at": "2026-06-02T00:00:00.000Z",
            "owner": {"uuid": NOVA_UUID, "handle": "nova", "name": "Nova Digital", "kind": "ai"},
            "participants": [],
        },
        "items": [{"uuid": f"019e7751-0000-7000-8000-00000000000{i}"} for i in range(items)],
    }


def problem(*, status, code, detail):
    """An RFC 9457 problem document — what a refused action returns."""
    return {
        "status": status,
        "code": code,
        "title": code.replace("_", " ").title(),
        "detail": detail,
    }


@pytest.fixture
def client():
    c = BaseCradle(token=FAKE_TOKEN)
    yield c
    c.close()


@pytest.fixture
def lock(client):
    """A LockTool bound to Nova's current timeline through a real client."""
    t = LockTool()
    t.bind(PlatformContext(client=client, timeline=TIMELINE_UUID))
    return t


def test_confirm_uuid_freezes_the_current_timeline_and_says_it_is_one_way(lock):
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/timelines/{TIMELINE_UUID}").mock(
            return_value=httpx.Response(200, json=timeline_envelope())
        )
        route = mock.post(f"{BC_URL}/timelines/{TIMELINE_UUID}/lock").mock(
            return_value=httpx.Response(200, json={"locked": True})
        )
        result = lock.run(confirm=TIMELINE_UUID)

    assert route.called
    assert "Locked timeline" in result
    assert "one-way" in result
    assert "operator-only" in result


def test_a_bare_call_previews_what_would_be_frozen_and_never_locks(lock):
    """B1 + preview-on-refuse: no confirm → a benign GET names the target, but no lock fires.

    Only the timeline GET is mocked; no lock route exists, so the gate must short-circuit
    before any destructive HTTP goes out — and the refusal must name the timeline + how to
    proceed with its uuid."""
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/timelines/{TIMELINE_UUID}").mock(
            return_value=httpx.Response(200, json=timeline_envelope(name="War Room", items=3))
        )
        result = lock.run()

    assert "Refused to lock" in result
    assert "War Room" in result  # the preview names what would be affected
    assert "3 item(s)" in result  # ...and how much is at stake
    assert "list, leave, or delete" in result  # names what it is NOT for
    assert f"confirm={TIMELINE_UUID}" in result  # the exact uuid to re-call with


def test_a_mismatched_confirm_is_refused_with_a_preview(lock):
    """The old boolean confirm=true no longer locks — it does not match the uuid, so it previews.

    This is the wrong-target gap the boolean left open: a confirm aimed at nothing (or the
    wrong room) must not lock the current one."""
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/timelines/{TIMELINE_UUID}").mock(
            return_value=httpx.Response(200, json=timeline_envelope())
        )
        result = lock.run(confirm="true")

    assert "Refused to lock" in result
    assert "'true'" in result  # the refusal echoes the non-matching confirm it got
    assert f"confirm={TIMELINE_UUID}" in result


def test_confirm_uuid_can_target_an_explicit_timeline(lock):
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/timelines/{OTHER_TIMELINE}").mock(
            return_value=httpx.Response(200, json=timeline_envelope(uuid=OTHER_TIMELINE))
        )
        route = mock.post(f"{BC_URL}/timelines/{OTHER_TIMELINE}/lock").mock(
            return_value=httpx.Response(200, json={"locked": True})
        )
        lock.run(confirm=OTHER_TIMELINE, timeline=OTHER_TIMELINE)

    assert route.called  # it locked the explicit timeline, not the bound one


def test_confirm_must_match_the_targeted_timeline_not_the_current_one(lock):
    """Targeting an explicit timeline but confirming the current one's uuid is a mismatch."""
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/timelines/{OTHER_TIMELINE}").mock(
            return_value=httpx.Response(200, json=timeline_envelope(uuid=OTHER_TIMELINE))
        )
        result = lock.run(confirm=TIMELINE_UUID, timeline=OTHER_TIMELINE)

    assert "Refused to lock" in result
    assert f"confirm={OTHER_TIMELINE}" in result  # confirm must match the TARGET


def test_a_platform_refusal_comes_back_as_a_clean_explanation(lock):
    detail = "This timeline is already locked."
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/timelines/{TIMELINE_UUID}").mock(
            return_value=httpx.Response(200, json=timeline_envelope())
        )
        mock.post(f"{BC_URL}/timelines/{TIMELINE_UUID}/lock").mock(
            return_value=httpx.Response(
                422, json=problem(status=422, code="already_locked", detail=detail)
            )
        )
        result = lock.run(confirm=TIMELINE_UUID)

    assert "Couldn't lock the timeline" in result
    assert detail in result  # the server's own human explanation is relayed verbatim


def test_unbound_tool_raises_platform_error():
    with pytest.raises(PlatformError):
        LockTool().run(confirm=TIMELINE_UUID)
