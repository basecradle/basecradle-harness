"""The standalone, confirm-guarded `LockTool`, against a respx-mocked platform.

Lock is the one irreversible platform action. Group 2b pulls it out of the
`timelines` tool into its own tool so a benign management call can never grab it by
accident (finding B1), behind a deliberate `confirm=true`. The headline invariants:
a bare call (no confirm) never touches the platform, and `confirm=true` actually
locks — the current timeline by default, or an explicit one.

Mocked at the transport level (a real `BaseCradle` client, only its HTTP stubbed),
per the constitution. Cast: Nova Digital (``nova``, AI) freezing her own room.
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


def timeline_envelope(*, uuid=TIMELINE_UUID, name="Incident response", locked=False):
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
        "items": [],
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


def test_confirm_true_freezes_the_current_timeline_and_says_it_is_one_way(lock):
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/timelines/{TIMELINE_UUID}").mock(
            return_value=httpx.Response(200, json=timeline_envelope())
        )
        route = mock.post(f"{BC_URL}/timelines/{TIMELINE_UUID}/lock").mock(
            return_value=httpx.Response(200, json={"locked": True})
        )
        result = lock.run(confirm=True)

    assert route.called
    assert "Locked timeline" in result
    assert "one-way" in result
    assert "operator-only" in result


def test_without_confirm_it_is_refused_and_never_touches_the_platform(lock):
    """B1: a bare call (no confirm) is refused locally — the irreversible op never fires.

    No lock route is mocked, so the guard must short-circuit before any HTTP goes out."""
    result = lock.run()

    assert "Refused to lock" in result
    assert "IRREVERSIBLE" in result
    assert "list, leave, or delete" in result  # names what it is NOT for
    assert "confirm=true" in result  # tells the model exactly how to proceed


def test_confirm_false_is_also_refused(lock):
    """An explicit confirm=false is not a deliberate yes — refused, nothing happens."""
    assert "Refused to lock" in lock.run(confirm=False)


def test_confirm_true_can_target_an_explicit_timeline(lock):
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/timelines/{OTHER_TIMELINE}").mock(
            return_value=httpx.Response(200, json=timeline_envelope(uuid=OTHER_TIMELINE))
        )
        route = mock.post(f"{BC_URL}/timelines/{OTHER_TIMELINE}/lock").mock(
            return_value=httpx.Response(200, json={"locked": True})
        )
        lock.run(confirm=True, timeline=OTHER_TIMELINE)

    assert route.called  # it locked the explicit timeline, not the bound one


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
        result = lock.run(confirm=True)

    assert "Couldn't lock the timeline" in result
    assert detail in result  # the server's own human explanation is relayed verbatim


def test_unbound_tool_raises_platform_error():
    with pytest.raises(PlatformError):
        LockTool().run(confirm=True)
