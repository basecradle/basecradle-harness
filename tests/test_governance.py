"""The governance tools (timelines + trust), against a respx-mocked platform.

No live platform call: respx stands in for the SDK's HTTP transport, returning
wire-shaped payloads (a timeline envelope, the user directory, a trust subject).
The fictional cast: Nova Digital (``nova``, AI) is the agent; John Doe (``john``,
human) is the peer she governs into and out of her timelines and trust graph.

The tools run through a *real* `BaseCradle` client — only its HTTP is mocked — so
these tests pin the tools against the SDK's true surface, the way the constitution
asks (mock at the transport level, never the live API). The headline invariants:
a refused action (not owner, no mutual trust) comes back as a clean explanation,
and lock is one-way (there is no unlock to test, by design).
"""

import json

import httpx
import pytest
import respx
from basecradle import BaseCradle

from basecradle_harness import PlatformContext, PlatformError, TimelinesTool, TrustTool
from basecradle_harness._governance import _resolve_user, _UserNotFound

BC_URL = "https://basecradle.com"
FAKE_TOKEN = "bc_uat_KqI8zFxkQ0OZ8vYwT7mWcVtR3nSdLpEa"

NOVA_UUID = "019e7750-66ee-79c8-ad8a-bbb6ea7c2bcc"  # the agent
JOHN_UUID = "019e7750-66ee-7e50-9e54-3bf8c3d6a8f1"  # the peer she governs
TIMELINE_UUID = "019e7750-66ee-7f53-829f-13a8a710b6da"  # the current timeline
NEW_TIMELINE_UUID = "019e7755-1a2b-7c3d-8e4f-0a1b2c3d4e5f"  # a freshly created one
OTHER_TIMELINE = "019e7760-1234-7abc-8def-0123456789ab"  # an explicit override


# --- wire payload builders ---------------------------------------------------


def user(
    *,
    uuid=JOHN_UUID,
    handle="john",
    name="John Doe",
    kind="human",
    you_trust=False,
    trusts_you=False,
):
    """A user in directory/subject form, with a trust cluster the tools render."""
    return {
        "uuid": uuid,
        "handle": handle,
        "name": name,
        "kind": kind,
        "trust": {
            "you_trust": you_trust,
            "trusts_you": trusts_you,
            "mutual": you_trust and trusts_you,
        },
    }


def directory(*users):
    """The user directory envelope (GET /users), defaulting to just John."""
    return {"users": list(users) if users else [user()]}


def timeline_envelope(
    *, uuid=TIMELINE_UUID, name="Incident response", locked=False, participants=None
):
    """The subject-timeline envelope the SDK merges on get/create."""
    if participants is None:
        participants = [{"uuid": NOVA_UUID, "handle": "nova", "name": "Nova Digital", "kind": "ai"}]
    return {
        "timeline": {
            "uuid": uuid,
            "name": name,
            "locked": locked,
            "created_at": "2026-06-01T00:00:00.000Z",
            "updated_at": "2026-06-02T00:00:00.000Z",
            "owner": {"uuid": NOVA_UUID, "handle": "nova", "name": "Nova Digital", "kind": "ai"},
            "participants": participants,
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
def timelines(client):
    """A TimelinesTool bound to Nova's current timeline through a real client."""
    t = TimelinesTool()
    t.bind(PlatformContext(client=client, timeline=TIMELINE_UUID))
    return t


@pytest.fixture
def trust(client):
    """A TrustTool bound through a real client."""
    t = TrustTool()
    t.bind(PlatformContext(client=client, timeline=TIMELINE_UUID))
    return t


# --- timelines: create -------------------------------------------------------


def test_create_makes_a_timeline_and_reports_its_uuid(timelines):
    captured = {}

    def capture(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(201, json=timeline_envelope(uuid=NEW_TIMELINE_UUID, name="Roadmap"))

    with respx.mock(assert_all_called=True) as mock:
        mock.post(f"{BC_URL}/timelines").mock(side_effect=capture)
        result = timelines.run(action="create", name="Roadmap")

    assert captured["body"] == {"timeline": {"name": "Roadmap"}}
    assert "Created timeline 'Roadmap'" in result
    assert NEW_TIMELINE_UUID in result
    assert "You own it" in result


def test_create_without_a_name_is_a_friendly_error(timelines):
    # No routes mocked: a missing name must fail before any request goes out.
    assert "needs a 'name'" in timelines.run(action="create")


# --- timelines: lock ---------------------------------------------------------


def test_lock_freezes_the_current_timeline_and_says_it_is_one_way(timelines):
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/timelines/{TIMELINE_UUID}").mock(
            return_value=httpx.Response(200, json=timeline_envelope())
        )
        lock = mock.post(f"{BC_URL}/timelines/{TIMELINE_UUID}/lock").mock(
            return_value=httpx.Response(200, json={"locked": True})
        )
        # Confirm echoes the exact uuid of the timeline being frozen — the deliberate guard.
        result = timelines.run(action="lock", confirm=TIMELINE_UUID)

    assert lock.called
    assert "Locked timeline" in result
    assert "one-way" in result
    assert "operator-only" in result


def test_lock_without_confirm_is_refused_and_never_touches_the_platform(timelines):
    """B1: a bare lock (no confirm) is refused locally — the irreversible op never fires.

    This is the live failure: a model wanting to list/delete a timeline grabbed `lock`.
    No lock route is mocked, so the guard must short-circuit before any HTTP goes out."""
    result = timelines.run(action="lock")

    assert "Refused to lock" in result
    assert "IRREVERSIBLE" in result
    assert "list, leave, or delete" in result  # names what it is NOT for
    assert TIMELINE_UUID in result  # tells the model the exact confirm value to use


def test_lock_with_a_wrong_confirm_is_refused(timelines):
    """A confirm that does not match the target uuid is not a deliberate echo — refused."""
    result = timelines.run(action="lock", confirm="not-the-uuid")

    assert "Refused to lock" in result


def test_lock_can_target_an_explicit_timeline(timelines):
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/timelines/{OTHER_TIMELINE}").mock(
            return_value=httpx.Response(200, json=timeline_envelope(uuid=OTHER_TIMELINE))
        )
        lock = mock.post(f"{BC_URL}/timelines/{OTHER_TIMELINE}/lock").mock(
            return_value=httpx.Response(200, json={"locked": True})
        )
        # Confirm must echo the explicit target, not the bound timeline.
        timelines.run(action="lock", timeline=OTHER_TIMELINE, confirm=OTHER_TIMELINE)

    assert lock.called  # it locked the explicit timeline, not the bound one


# --- timelines: add / remove participant -------------------------------------


def test_add_participant_resolves_a_handle_and_adds_them(timelines):
    captured = {}

    def capture(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(201, json=user())

    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/users").mock(return_value=httpx.Response(200, json=directory()))
        mock.get(f"{BC_URL}/timelines/{TIMELINE_UUID}").mock(
            return_value=httpx.Response(200, json=timeline_envelope())
        )
        mock.post(f"{BC_URL}/timelines/{TIMELINE_UUID}/participations").mock(side_effect=capture)
        result = timelines.run(action="add_participant", user="@john")

    assert captured["body"] == {"user_id": JOHN_UUID}  # the handle resolved to a uuid
    assert "Added @john" in result


def test_add_participant_surfaces_a_missing_mutual_trust_as_a_clean_explanation(timelines):
    detail = "You and @john do not have mutual trust, so they can't join this timeline."
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/users").mock(return_value=httpx.Response(200, json=directory()))
        mock.get(f"{BC_URL}/timelines/{TIMELINE_UUID}").mock(
            return_value=httpx.Response(200, json=timeline_envelope())
        )
        mock.post(f"{BC_URL}/timelines/{TIMELINE_UUID}/participations").mock(
            return_value=httpx.Response(
                422, json=problem(status=422, code="validation_failed", detail=detail)
            )
        )
        result = timelines.run(action="add_participant", user="john")

    assert "Couldn't add the participant" in result
    assert detail in result  # the server's own human explanation is relayed verbatim


def test_add_participant_surfaces_a_not_owner_refusal(timelines):
    detail = "The action requires being the timeline's owner."
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/users").mock(return_value=httpx.Response(200, json=directory()))
        mock.get(f"{BC_URL}/timelines/{TIMELINE_UUID}").mock(
            return_value=httpx.Response(200, json=timeline_envelope())
        )
        mock.post(f"{BC_URL}/timelines/{TIMELINE_UUID}/participations").mock(
            return_value=httpx.Response(
                403, json=problem(status=403, code="not_timeline_owner", detail=detail)
            )
        )
        result = timelines.run(action="add_participant", user="john")

    assert "Couldn't add the participant" in result
    assert detail in result


def test_add_participant_with_an_unknown_handle_fails_before_touching_the_timeline(timelines):
    with respx.mock(assert_all_called=True) as mock:
        # Only the directory is consulted; no timeline/participations call is made.
        mock.get(f"{BC_URL}/users").mock(return_value=httpx.Response(200, json=directory()))
        result = timelines.run(action="add_participant", user="@ghost")

    assert "No user with handle '@ghost'" in result


def test_add_participant_without_a_user_is_a_friendly_error(timelines):
    assert "needs a 'user'" in timelines.run(action="add_participant")


def test_remove_participant_resolves_a_uuid_and_removes_them(timelines):
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/users/{JOHN_UUID}").mock(
            return_value=httpx.Response(200, json={"user": user()})
        )
        mock.get(f"{BC_URL}/timelines/{TIMELINE_UUID}").mock(
            return_value=httpx.Response(200, json=timeline_envelope())
        )
        route = mock.delete(f"{BC_URL}/timelines/{TIMELINE_UUID}/participations/{JOHN_UUID}").mock(
            return_value=httpx.Response(204)
        )
        result = timelines.run(action="remove_participant", user=JOHN_UUID)

    assert route.called
    assert "Removed @john" in result


# --- timelines: dispatch / binding -------------------------------------------


def test_timelines_unknown_action_is_reported(timelines):
    assert "unknown action" in timelines.run(action="archive")


def test_timelines_unbound_tool_raises_platform_error():
    with pytest.raises(PlatformError):
        TimelinesTool().run(action="lock")


# --- trust: grant ------------------------------------------------------------


def test_grant_trust_by_handle_reports_not_yet_mutual(trust):
    granted = user(you_trust=True, trusts_you=False)  # Nova trusts John; he hasn't reciprocated
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/users").mock(return_value=httpx.Response(200, json=directory()))
        route = mock.post(f"{BC_URL}/users/{JOHN_UUID}/trust").mock(
            return_value=httpx.Response(200, json={"user": granted})
        )
        result = trust.run(action="grant", user="@john")

    assert route.called
    assert "You now trust @john" in result
    assert "not yet mutual" in result
    # B4: granting must not claim the reverse edge when it doesn't exist.
    assert "they trust you" not in result.lower()


def test_grant_trust_reports_mutual_when_reciprocated(trust):
    granted = user(you_trust=True, trusts_you=True)  # both edges present
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/users").mock(return_value=httpx.Response(200, json=directory()))
        mock.post(f"{BC_URL}/users/{JOHN_UUID}/trust").mock(
            return_value=httpx.Response(200, json={"user": granted})
        )
        result = trust.run(action="grant", user="john")

    # B4: mutuality is reported as a pre-existing reverse edge ("they already trusted you"),
    # framed as fact, not as a consequence of this directional grant.
    assert "now mutual" in result
    assert "already trusted you" in result


def test_grant_trust_by_uuid_skips_the_directory(trust):
    granted = user(you_trust=True)
    with respx.mock(assert_all_called=True) as mock:
        # Resolved straight by uuid — the directory is never scanned.
        mock.get(f"{BC_URL}/users/{JOHN_UUID}").mock(
            return_value=httpx.Response(200, json={"user": user()})
        )
        mock.post(f"{BC_URL}/users/{JOHN_UUID}/trust").mock(
            return_value=httpx.Response(200, json={"user": granted})
        )
        result = trust.run(action="grant", user=JOHN_UUID)

    assert "You now trust @john" in result


# --- trust: revoke -----------------------------------------------------------


def test_revoke_trust_removes_the_edge_and_explains_no_eviction(trust):
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/users").mock(return_value=httpx.Response(200, json=directory()))
        route = mock.delete(f"{BC_URL}/users/{JOHN_UUID}/trust").mock(
            return_value=httpx.Response(204)
        )
        result = trust.run(action="revoke", user="@john")

    assert route.called
    assert "You no longer trust @john" in result


def test_trust_with_an_unknown_handle_is_a_friendly_error(trust):
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/users").mock(return_value=httpx.Response(200, json=directory()))
        assert "No user with handle '@ghost'" in trust.run(action="grant", user="@ghost")


def test_trust_unknown_action_is_reported(trust):
    assert "unknown action" in trust.run(action="distrust", user="@john")


def test_trust_without_a_user_is_a_friendly_error(trust):
    assert "needs a 'user'" in trust.run(action="grant")


def test_trust_unbound_tool_raises_platform_error():
    with pytest.raises(PlatformError):
        TrustTool().run(action="grant", user="@john")


# --- user resolution (unit) --------------------------------------------------


def test_resolve_user_by_uuid_fetches_directly(client):
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/users/{JOHN_UUID}").mock(
            return_value=httpx.Response(200, json={"user": user()})
        )
        resolved = _resolve_user(client, JOHN_UUID)

    assert resolved.uuid == JOHN_UUID


@pytest.mark.parametrize("reference", ["john", "@john", "@JOHN", "JOHN"])
def test_resolve_user_by_handle_is_case_insensitive_and_at_optional(client, reference):
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/users").mock(
            return_value=httpx.Response(
                200, json=directory(user(uuid=NOVA_UUID, handle="nova", kind="ai"), user())
            )
        )
        resolved = _resolve_user(client, reference)

    assert resolved.uuid == JOHN_UUID


def test_resolve_user_raises_when_no_handle_matches(client):
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/users").mock(return_value=httpx.Response(200, json=directory()))
        with pytest.raises(_UserNotFound, match="ghost"):
            _resolve_user(client, "@ghost")
