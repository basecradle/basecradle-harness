"""The standalone `DeleteTool`, behind the shared uuid-confirm gate, against a respx platform.

Delete is the destructive owner power that restores human–AI parity (a human owner can delete
their timeline; the harnessed peer now can too). It is the second irreversible timeline action,
and shares the one `ConfirmedTimelineAction` gate with lock (see `_confirmed.py` and
`test_lock.py`) — they must behave identically at the gate. The headline invariants: a bare or
mismatched confirm never deletes (only a benign preview GET goes out), `confirm=<uuid>` actually
deletes (`DELETE /timelines/:uuid`, 204), and a not-owner refusal comes back as a clean
explanation.

Mocked at the transport level (a real `BaseCradle` client, only its HTTP stubbed), per the
constitution. Cast: Nova Digital (``nova``, AI) deleting her own room.
"""

import httpx
import pytest
import respx
from basecradle import BaseCradle

from basecradle_harness import DeleteTool, PlatformContext, PlatformError

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
def delete(client):
    """A DeleteTool bound to Nova's current timeline through a real client."""
    t = DeleteTool()
    t.bind(PlatformContext(client=client, timeline=TIMELINE_UUID))
    return t


def test_confirm_uuid_deletes_the_current_timeline_and_says_it_is_gone(delete):
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/timelines/{TIMELINE_UUID}").mock(
            return_value=httpx.Response(200, json=timeline_envelope(name="War Room"))
        )
        route = mock.delete(f"{BC_URL}/timelines/{TIMELINE_UUID}").mock(
            return_value=httpx.Response(204)
        )
        result = delete.run(confirm=TIMELINE_UUID)

    assert route.called  # DELETE /timelines/:uuid actually fired
    assert "Deleted timeline 'War Room'" in result
    assert "cannot be undone" in result


def test_a_bare_call_previews_what_would_be_destroyed_and_never_deletes(delete):
    """preview-on-refuse: no confirm → a benign GET names the target, but no DELETE fires.

    Only the timeline GET is mocked; no delete route exists, so the gate must short-circuit
    before any destructive HTTP goes out — and the refusal must name the timeline + how to
    proceed with its uuid."""
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/timelines/{TIMELINE_UUID}").mock(
            return_value=httpx.Response(200, json=timeline_envelope(name="War Room", items=5))
        )
        result = delete.run()

    assert "Refused to delete" in result
    assert "War Room" in result  # the preview names what would be affected
    assert "5 item(s)" in result  # ...and how much is at stake
    assert "all its content" in result  # names the destructive consequence
    assert f"confirm={TIMELINE_UUID}" in result  # the exact uuid to re-call with


def test_a_mismatched_confirm_is_refused_with_a_preview(delete):
    """A confirm that does not equal the target uuid must not delete — the wrong-target guard."""
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/timelines/{TIMELINE_UUID}").mock(
            return_value=httpx.Response(200, json=timeline_envelope())
        )
        result = delete.run(confirm="yes")

    assert "Refused to delete" in result
    assert "'yes'" in result  # the refusal echoes the non-matching confirm it got
    assert f"confirm={TIMELINE_UUID}" in result


def test_confirm_uuid_can_target_an_explicit_timeline(delete):
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/timelines/{OTHER_TIMELINE}").mock(
            return_value=httpx.Response(200, json=timeline_envelope(uuid=OTHER_TIMELINE))
        )
        route = mock.delete(f"{BC_URL}/timelines/{OTHER_TIMELINE}").mock(
            return_value=httpx.Response(204)
        )
        delete.run(confirm=OTHER_TIMELINE, timeline=OTHER_TIMELINE)

    assert route.called  # it deleted the explicit timeline, not the bound one


def test_confirm_must_match_the_targeted_timeline_not_the_current_one(delete):
    """Targeting an explicit timeline but confirming the current one's uuid is a mismatch."""
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/timelines/{OTHER_TIMELINE}").mock(
            return_value=httpx.Response(200, json=timeline_envelope(uuid=OTHER_TIMELINE))
        )
        result = delete.run(confirm=TIMELINE_UUID, timeline=OTHER_TIMELINE)

    assert "Refused to delete" in result
    assert f"confirm={OTHER_TIMELINE}" in result  # confirm must match the TARGET


def test_a_not_owner_refusal_comes_back_as_a_clean_explanation(delete):
    """Deletion is owner-or-admin only; a participant gets 403, relayed as the server's words."""
    detail = "The action requires being the timeline's owner."
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/timelines/{TIMELINE_UUID}").mock(
            return_value=httpx.Response(200, json=timeline_envelope())
        )
        mock.delete(f"{BC_URL}/timelines/{TIMELINE_UUID}").mock(
            return_value=httpx.Response(
                403, json=problem(status=403, code="not_timeline_owner", detail=detail)
            )
        )
        result = delete.run(confirm=TIMELINE_UUID)

    assert "Couldn't delete the timeline" in result
    assert detail in result  # the server's own human explanation is relayed verbatim


def test_unbound_tool_raises_platform_error():
    with pytest.raises(PlatformError):
        DeleteTool().run(confirm=TIMELINE_UUID)
