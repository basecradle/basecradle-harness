"""The read tools (`users` + `messages`), against a respx-mocked platform.

These are the cure for **the blind peer** (B5): the reads that let an agent say who
is on the platform, what its trust with someone is, who is on a timeline, and what
was said before it woke. Mocked at the transport level (a real `BaseCradle` client,
only its HTTP stubbed), per the constitution. Cast: Nova Digital (``nova``, AI) is
the agent looking out; John Doe (``john``, human) is the peer she reads.

The headline invariants: `list` renders the trust state per user (the direct answer
to "what's my trust with everyone"); `read` surfaces only the profile fields the API
returned (access tiers are the platform's job — a withheld field is omitted, never
invented); and `messages` reads the backlog the wake didn't hand over.
"""

import httpx
import pytest
import respx
from basecradle import BaseCradle

from basecradle_harness import MessagesTool, PlatformContext, PlatformError, UsersTool

BC_URL = "https://basecradle.com"
FAKE_TOKEN = "bc_uat_KqI8zFxkQ0OZ8vYwT7mWcVtR3nSdLpEa"

NOVA_UUID = "019e7750-66ee-79c8-ad8a-bbb6ea7c2bcc"  # the agent
JOHN_UUID = "019e7750-66ee-7e50-9e54-3bf8c3d6a8f1"  # the peer she reads
TIMELINE_UUID = "019e7750-66ee-7f53-829f-13a8a710b6da"  # the current timeline
MESSAGE_UUID = "019e7751-1111-7222-8333-444455556666"


# --- wire payload builders ---------------------------------------------------


def user(
    *,
    uuid=JOHN_UUID,
    handle="john",
    name="John Doe",
    kind="human",
    you_trust=False,
    trusts_you=False,
    **extra,
):
    """A user in directory/subject form, with a trust cluster and any extra tier fields."""
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
        **extra,
    }


def directory(*users):
    """The user directory envelope (GET /users), defaulting to just John."""
    return {"users": list(users) if users else [user()]}


def dashboard():
    """A minimal but valid GET /users/dashboard envelope (what bc.me returns)."""
    return {
        "identity": user(uuid=NOVA_UUID, handle="nova", name="Nova Digital", kind="ai"),
        "environment": {
            "name": "BaseCradle",
            "summary": "A communications platform where humans and AI are peers.",
            "you_are": "a first-class peer here",
        },
        "interaction": {
            "timelines": {"url": f"{BC_URL}/timelines", "count": 3},
            "assets_url": f"{BC_URL}/assets",
            "messages_url": f"{BC_URL}/messages",
            "tasks_url": f"{BC_URL}/tasks",
            "webhook_endpoints_url": f"{BC_URL}/webhook_endpoints",
            "webhook_events_url": f"{BC_URL}/webhook_events",
        },
        "account": {
            "profile_url": f"{BC_URL}/profile",
            "sessions_url": f"{BC_URL}/sessions",
            "change_password_url": f"{BC_URL}/password",
        },
        "documentation": {
            "user_guide": f"{BC_URL}/guide",
            "api": f"{BC_URL}/docs/api",
            "changelog": f"{BC_URL}/changelog",
            "openapi": f"{BC_URL}/openapi.json",
            "reference": f"{BC_URL}/reference",
            "sdks": {"python": {"repository": "gh/python", "package": "basecradle"}},
        },
    }


def timeline_envelope(uuid=TIMELINE_UUID):
    """A GET /timelines/{uuid} envelope (the SDK merges its two keys into one Timeline)."""
    return {"timeline": {"uuid": uuid}, "items": []}


def problem(*, code, detail, status):
    """An RFC 9457 problem document (the SDK needs a `code` to map it to a typed error)."""
    return {
        "code": code,
        "title": code.replace("_", " ").title(),
        "detail": detail,
        "status": status,
    }


def message(*, uuid=MESSAGE_UUID, body="Heads up: the deploy is live.", handle="john"):
    """A message item envelope (Item shape: type, created_at, user, timeline, content)."""
    return {
        "type": "message",
        "created_at": "2026-06-15T12:00:00.000Z",
        "user": {"uuid": JOHN_UUID, "handle": handle, "name": "John Doe", "kind": "human"},
        "timeline": {"uuid": TIMELINE_UUID},
        "content": {"uuid": uuid, "body": body},
    }


@pytest.fixture
def client():
    c = BaseCradle(token=FAKE_TOKEN)
    yield c
    c.close()


@pytest.fixture
def users(client):
    """A UsersTool bound through a real client."""
    t = UsersTool()
    t.bind(PlatformContext(client=client, timeline=TIMELINE_UUID))
    return t


@pytest.fixture
def messages(client):
    """A MessagesTool bound to Nova's current timeline through a real client."""
    t = MessagesTool()
    t.bind(PlatformContext(client=client, timeline=TIMELINE_UUID))
    return t


# --- users: list -------------------------------------------------------------


def test_list_renders_the_directory_with_trust_state(users):
    people = directory(
        user(uuid=JOHN_UUID, handle="john", you_trust=True, trusts_you=True),  # mutual
        user(uuid="019e7752-aaaa-7bbb-8ccc-ddddeeeeffff", handle="origin", kind="ai"),
    )
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/users").mock(return_value=httpx.Response(200, json=people))
        result = users.run(action="list")

    assert "@john" in result and "@origin" in result
    assert "mutual trust" in result  # the trust state per user — the B4 cure
    assert "no trust either way" in result


def test_list_with_an_empty_directory_says_so(users):
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/users").mock(return_value=httpx.Response(200, json={"users": []}))
        assert "No other users" in users.run(action="list")


# --- users: read -------------------------------------------------------------


def test_read_by_uuid_surfaces_richer_profile_fields_when_present(users):
    subject = user(
        you_trust=True,
        about="Incident commander.",
        time_zone="America/Chicago",
        roles=["admin"],
    )
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/users/{JOHN_UUID}").mock(
            return_value=httpx.Response(200, json={"user": subject})
        )
        result = users.run(action="read", user=JOHN_UUID)

    assert "@john" in result
    assert "you trust them; not reciprocated" in result
    assert "About: Incident commander." in result
    assert "Time zone: America/Chicago" in result
    assert "Roles: admin" in result


def test_read_omits_access_gated_fields_the_api_withheld(users):
    """A directory-tier view carries only base identity + trust — no about/roles to crash on."""
    with respx.mock(assert_all_called=True) as mock:
        # Resolved by handle → scans the directory, which carries the base view only.
        mock.get(f"{BC_URL}/users").mock(return_value=httpx.Response(200, json=directory()))
        result = users.run(action="read", user="@john")

    assert "@john" in result
    assert "Trust:" in result
    assert "About:" not in result  # withheld → omitted, never invented
    assert "Roles:" not in result


def test_read_without_a_user_is_a_friendly_error(users):
    assert "needs a 'user'" in users.run(action="read")


def test_read_unknown_handle_is_a_friendly_error(users):
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/users").mock(return_value=httpx.Response(200, json=directory()))
        assert "No user with handle '@ghost'" in users.run(action="read", user="@ghost")


# --- users: me (dashboard) ---------------------------------------------------


def test_me_returns_the_dashboard_identity_and_environment(users):
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/users/dashboard").mock(
            return_value=httpx.Response(200, json=dashboard())
        )
        result = users.run(action="me")

    assert "@nova" in result and "Nova Digital" in result
    assert "first-class peer" in result
    assert "Your timelines: 3" in result


# --- users: dispatch / binding -----------------------------------------------


def test_users_unknown_action_is_reported(users):
    assert "unknown action" in users.run(action="search")


def test_users_unbound_tool_raises_platform_error():
    with pytest.raises(PlatformError):
        UsersTool().run(action="list")


# --- messages: list ----------------------------------------------------------


def test_messages_list_shows_recent_messages_newest_first(messages):
    page = {
        "messages": [
            message(body="Second."),
            message(uuid="019e7751-0000-7000-8000-000000000001", body="First."),
        ],
        "next_cursor": None,
    }
    captured = {}

    def capture(request):
        captured["url"] = str(request.url)
        return httpx.Response(200, json=page)

    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/messages").mock(side_effect=capture)
        result = messages.run(action="list")

    assert f"timeline={TIMELINE_UUID}" in captured["url"]  # filtered to the current timeline
    assert "Second." in result and "First." in result
    assert "@john" in result


def test_messages_list_with_no_messages_says_so(messages):
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/messages").mock(
            return_value=httpx.Response(200, json={"messages": [], "next_cursor": None})
        )
        assert "No messages on this timeline" in messages.run(action="list")


# --- messages: read ----------------------------------------------------------


def test_messages_read_returns_one_message_in_full(messages):
    full = "Heads up: the deploy is live and the smoke tests are green across every region."
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/messages/{MESSAGE_UUID}").mock(
            return_value=httpx.Response(200, json={"message": message(body=full)})
        )
        result = messages.run(action="read", uuid=MESSAGE_UUID)

    assert full in result  # the whole body, not a preview
    assert MESSAGE_UUID in result


def test_messages_read_without_a_uuid_is_a_friendly_error(messages):
    assert "needs the message's uuid" in messages.run(action="read")


# --- messages: create --------------------------------------------------------


def test_messages_create_posts_to_the_current_timeline_and_returns_the_uuid(messages):
    """Omitting `timeline` posts to the wake-timeline; the new uuid comes back."""
    new_uuid = "019e7755-aaaa-7bbb-8ccc-ddddeeee0001"
    captured = {}

    def capture(request):
        captured["url"] = str(request.url)
        captured["body"] = request.content
        return httpx.Response(201, json={"message": message(uuid=new_uuid, handle="nova")})

    with respx.mock(assert_all_called=True) as mock:
        # The SDK's nested creator resolves the timeline (a GET) then POSTs the message.
        mock.get(f"{BC_URL}/timelines/{TIMELINE_UUID}").mock(
            return_value=httpx.Response(200, json=timeline_envelope())
        )
        mock.post(f"{BC_URL}/timelines/{TIMELINE_UUID}/messages").mock(side_effect=capture)
        result = messages.run(action="create", body="Heads up: I hit a bug on the build.")

    assert f"/timelines/{TIMELINE_UUID}/messages" in captured["url"]  # the current timeline
    assert b"Heads up: I hit a bug on the build." in captured["body"]
    assert new_uuid in result  # the created message's uuid, for the model to relay


def test_messages_create_posts_cross_timeline_by_uuid(messages):
    """An explicit `timeline` posts to a *different* viewable timeline — the support pattern."""
    support = "019e7760-bbbb-7ccc-8ddd-eeeeffff0002"
    new_uuid = "019e7755-aaaa-7bbb-8ccc-ddddeeee0003"

    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/timelines/{support}").mock(
            return_value=httpx.Response(200, json=timeline_envelope(support))
        )
        mock.post(f"{BC_URL}/timelines/{support}/messages").mock(
            return_value=httpx.Response(
                201, json={"message": message(uuid=new_uuid, handle="nova")}
            )
        )
        result = messages.run(
            action="create", body="Escalating from the project room.", timeline=support
        )

    assert new_uuid in result  # posted to the support timeline, not the current one


def test_messages_create_relays_a_locked_timeline_refusal_without_retrying(messages):
    """A locked timeline rejects the post — relayed cleanly, and the POST is made exactly once."""
    posts = {"count": 0}

    def refuse(request):
        posts["count"] += 1
        return httpx.Response(
            403,
            json=problem(code="timeline_locked", detail="This timeline is locked.", status=403),
        )

    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/timelines/{TIMELINE_UUID}").mock(
            return_value=httpx.Response(200, json=timeline_envelope())
        )
        mock.post(f"{BC_URL}/timelines/{TIMELINE_UUID}/messages").mock(side_effect=refuse)
        result = messages.run(action="create", body="Anyone there?")

    assert "This timeline is locked." in result  # the platform's actual reason, relayed
    assert "Couldn't create a message" in result
    assert posts["count"] == 1  # no blind retry — a double-post would wake the recipient twice


def test_messages_create_relays_a_not_a_viewer_refusal(messages):
    """Posting to a timeline you can't view is refused server-side — relayed, not crashed."""
    other = "019e7761-cccc-7ddd-8eee-ffff00000004"

    with respx.mock(assert_all_called=True) as mock:
        # The timeline resolve itself is refused — you are not a viewer.
        mock.get(f"{BC_URL}/timelines/{other}").mock(
            return_value=httpx.Response(
                403,
                json=problem(
                    code="not_a_viewer",
                    detail="You are not a viewer of this timeline.",
                    status=403,
                ),
            )
        )
        result = messages.run(action="create", body="Hello?", timeline=other)

    assert "You are not a viewer of this timeline." in result
    assert "Couldn't create a message" in result


def test_messages_create_without_a_body_is_a_friendly_error(messages):
    assert "needs a 'body'" in messages.run(action="create")


def test_messages_unknown_action_is_reported(messages):
    assert "unknown action" in messages.run(action="delete")


def test_messages_unbound_tool_raises_platform_error():
    with pytest.raises(PlatformError):
        MessagesTool().run(action="list")
