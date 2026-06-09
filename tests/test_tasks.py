"""The tasks tool, against a respx-mocked platform.

No live platform call: respx stands in for the SDK's HTTP transport, returning
wire-shaped task payloads (list page, single task, create). The fictional cast:
Nova Digital (``nova``, AI) is the agent; tasks live on John Doe's timeline.

The tool is exercised through a *real* `BaseCradle` client — only its HTTP is
mocked — so these tests pin the tool against the SDK's true surface, the way the
constitution asks (mock at the transport level, never the live API).
"""

import json
from datetime import datetime, timezone

import httpx
import pytest
import respx
from basecradle import BaseCradle

from basecradle_harness import PlatformContext, PlatformError, TasksTool
from basecradle_harness._tasks import _normalize_activate_at

BC_URL = "https://basecradle.com"
FAKE_TOKEN = "bc_uat_KqI8zFxkQ0OZ8vYwT7mWcVtR3nSdLpEa"

NOVA_UUID = "019e7750-66ee-79c8-ad8a-bbb6ea7c2bcc"  # the agent
JOHN_UUID = "019e7750-66ee-7e50-9e54-3bf8c3d6a8f1"  # the human
TIMELINE_UUID = "019e7750-66ee-7f53-829f-13a8a710b6da"
OTHER_TIMELINE = "019e7760-1234-7abc-8def-0123456789ab"

# Well-formed UUIDv7 task (content) uuids.
T_ONE = "019e7751-4a1b-7c2d-8e3f-1a2b3c4d5e6f"
T_TWO = "019e7752-5b2c-7d3e-9f40-2b3c4d5e6f70"

# A fixed "now" so relative offsets resolve deterministically.
FIXED_NOW = datetime(2026, 6, 9, 12, 0, 0, tzinfo=timezone.utc)


# --- wire payload builders ---------------------------------------------------


def task(*, uuid, instructions, activate_at="2026-06-10T15:00:00Z", status="pending"):
    """A task in subject form (the SDK's documented shape)."""
    return {
        "type": "task",
        "created_at": "2026-06-04T00:00:00.000Z",
        "user": {"uuid": NOVA_UUID, "handle": "nova", "name": "Nova Digital", "kind": "ai"},
        "timeline": {"uuid": TIMELINE_UUID},
        "content": {
            "uuid": uuid,
            "instructions": instructions,
            "activate_at": activate_at,
            "status": status,
        },
    }


def a_task():
    return task(uuid=T_ONE, instructions="Summarize the incident thread", status="pending")


def another_task():
    return task(uuid=T_TWO, instructions="Post the daily digest", status="activated")


def _numbered_task(i):
    """A distinct, well-formed task payload for list-cap tests (varied node bits)."""
    node = (i * 0x9E3779B1) & 0xFFFFFFFFFFFF
    return task(uuid=f"019e7758-1a2b-7c3d-8e4f-{node:012x}", instructions=f"Task {i}")


@pytest.fixture
def client():
    c = BaseCradle(token=FAKE_TOKEN)
    yield c
    c.close()


@pytest.fixture
def tool(client):
    """A TasksTool bound to John's timeline through a real client."""
    t = TasksTool()
    t.bind(PlatformContext(client=client, timeline=TIMELINE_UUID))
    return t


# --- create ------------------------------------------------------------------


def test_create_with_a_relative_offset_normalizes_against_now(tool, monkeypatch):
    monkeypatch.setattr("basecradle_harness._tasks._utcnow", lambda: FIXED_NOW)
    captured = {}

    def capture(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(201, json={"task": a_task()})

    with respx.mock(assert_all_called=True) as mock:
        # create resolves the timeline (public path), then POSTs the task.
        mock.get(f"{BC_URL}/timelines/{TIMELINE_UUID}").mock(
            return_value=httpx.Response(200, json=_timeline_envelope())
        )
        mock.post(f"{BC_URL}/timelines/{TIMELINE_UUID}/tasks").mock(side_effect=capture)
        result = tool.run(
            action="create",
            instructions="Summarize the incident thread",
            activate_at="+2h",
        )

    assert "Scheduled a task" in result
    # +2h from a fixed noon resolves to 14:00 UTC, sent as an absolute timestamp.
    assert captured["body"]["task"]["activate_at"] == "2026-06-09T14:00:00+00:00"
    assert captured["body"]["task"]["instructions"] == "Summarize the incident thread"


def test_create_with_an_absolute_iso_timestamp_passes_it_through(tool):
    captured = {}

    def capture(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(201, json={"task": a_task()})

    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/timelines/{TIMELINE_UUID}").mock(
            return_value=httpx.Response(200, json=_timeline_envelope())
        )
        mock.post(f"{BC_URL}/timelines/{TIMELINE_UUID}/tasks").mock(side_effect=capture)
        tool.run(
            action="create",
            instructions="Post the daily digest",
            activate_at="2026-06-10T15:00:00Z",
        )

    # The trailing 'Z' is normalized to the equivalent offset; the instant is unchanged.
    assert captured["body"]["task"]["activate_at"] == "2026-06-10T15:00:00+00:00"


def test_create_requires_instructions_and_activate_at(tool):
    assert "needs both" in tool.run(action="create", instructions="do a thing")
    assert "needs both" in tool.run(action="create", activate_at="+1h")


def test_create_rejects_an_unparseable_activate_at_without_calling_the_api(tool):
    # No HTTP routes mocked: a bad time must fail before any request goes out.
    result = tool.run(action="create", instructions="do a thing", activate_at="next thursday-ish")
    assert "Error" in result
    assert "ISO-8601" in result


# --- list --------------------------------------------------------------------


def test_list_renders_each_task_with_its_uuid_status_and_time(tool):
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/tasks", params={"timeline": TIMELINE_UUID}).mock(
            return_value=httpx.Response(
                200, json={"tasks": [a_task(), another_task()], "next_cursor": None}
            )
        )
        result = tool.run(action="list")

    assert T_ONE in result
    assert "Summarize the incident thread" in result
    assert "pending" in result
    assert "activated" in result
    assert "2026-06-10T15:00:00Z" in result


def test_list_on_an_empty_timeline_says_so(tool):
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/tasks").mock(
            return_value=httpx.Response(200, json={"tasks": [], "next_cursor": None})
        )
        assert "No tasks" in tool.run(action="list")


def test_list_truncates_long_instructions_in_the_preview(tool):
    long = "x" * 500
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/tasks").mock(
            return_value=httpx.Response(
                200,
                json={"tasks": [task(uuid=T_ONE, instructions=long)], "next_cursor": None},
            )
        )
        result = tool.run(action="list")

    assert "…" in result  # elided
    assert long not in result  # the full body is not dumped into a list


def test_list_does_not_claim_more_at_exactly_the_limit(tool, monkeypatch):
    monkeypatch.setattr("basecradle_harness._tasks.DEFAULT_LIST_LIMIT", 2)
    two = [_numbered_task(0), _numbered_task(1)]
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/tasks").mock(
            return_value=httpx.Response(200, json={"tasks": two, "next_cursor": None})
        )
        result = tool.run(action="list")

    assert "there may be more" not in result


def test_list_claims_more_past_the_limit(tool, monkeypatch):
    monkeypatch.setattr("basecradle_harness._tasks.DEFAULT_LIST_LIMIT", 2)
    three = [_numbered_task(i) for i in range(3)]
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/tasks").mock(
            return_value=httpx.Response(200, json={"tasks": three, "next_cursor": None})
        )
        result = tool.run(action="list")

    assert "there may be more" in result
    assert _numbered_task(2)["content"]["uuid"] not in result


# --- read --------------------------------------------------------------------


def test_read_returns_the_full_task(tool):
    full = "Summarize the incident thread and post the digest to the timeline."
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/tasks/{T_ONE}").mock(
            return_value=httpx.Response(200, json={"task": task(uuid=T_ONE, instructions=full)})
        )
        result = tool.run(action="read", uuid=T_ONE)

    assert full in result  # the whole body, not a preview
    assert T_ONE in result
    assert "pending" in result


def test_read_without_a_uuid_is_a_friendly_error(tool):
    assert "needs the task's uuid" in tool.run(action="read")


# --- cross-timeline + validation + binding -----------------------------------


def test_an_explicit_timeline_overrides_the_current_one(tool):
    with respx.mock(assert_all_called=True) as mock:
        route = mock.get(f"{BC_URL}/tasks", params={"timeline": OTHER_TIMELINE}).mock(
            return_value=httpx.Response(200, json={"tasks": [], "next_cursor": None})
        )
        tool.run(action="list", timeline=OTHER_TIMELINE)

    assert route.called  # it queried the other timeline, not the bound one


def test_unknown_action_is_reported(tool):
    assert "unknown action" in tool.run(action="delete")


def test_an_unbound_tool_raises_platform_error():
    with pytest.raises(PlatformError):
        TasksTool().run(action="list")


# --- activate_at normalization (unit) ----------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("+30s", "2026-06-09T12:00:30+00:00"),
        ("+15m", "2026-06-09T12:15:00+00:00"),
        ("+2h", "2026-06-09T14:00:00+00:00"),
        ("+1d", "2026-06-10T12:00:00+00:00"),
        ("+1w", "2026-06-16T12:00:00+00:00"),
        ("+ 2 h", "2026-06-09T14:00:00+00:00"),  # whitespace tolerated
    ],
)
def test_normalize_relative_offsets(value, expected, monkeypatch):
    monkeypatch.setattr("basecradle_harness._tasks._utcnow", lambda: FIXED_NOW)
    assert _normalize_activate_at(value) == expected


def test_normalize_absolute_with_trailing_z():
    assert _normalize_activate_at("2026-06-10T15:00:00Z") == "2026-06-10T15:00:00+00:00"


def test_normalize_absolute_naive_is_read_as_utc():
    assert _normalize_activate_at("2026-06-10T15:00:00") == "2026-06-10T15:00:00+00:00"


def test_normalize_absolute_with_explicit_offset_is_preserved():
    assert _normalize_activate_at("2026-06-10T15:00:00-05:00") == "2026-06-10T15:00:00-05:00"


@pytest.mark.parametrize("bad", ["soon", "next week", "+2", "+2y", "2026-13-99", ""])
def test_normalize_rejects_garbage(bad):
    with pytest.raises(ValueError, match="ISO-8601"):
        _normalize_activate_at(bad)


# --- shared wire helper ------------------------------------------------------


def _timeline_envelope():
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
