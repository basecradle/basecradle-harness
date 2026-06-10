"""The webhook tools (endpoints + events), against a respx-mocked platform.

No live platform call: respx stands in for the SDK's HTTP transport, returning
wire-shaped payloads (an endpoint subject, a paginated list, an event subject).
The tools run through a *real* `BaseCradle` client — only its HTTP is mocked — so
these tests pin the tools against the SDK's true surface, the way the constitution
asks (mock at the transport level, never the live API).

The fictional cast: Nova Digital (``nova``, AI) is the agent running endpoints on
her timeline; the inbound deliveries arrive from an external sender. The headline
invariants: `create` surfaces the secret ingest URL, `rotate` reports the new one
and kills the old, a refused action comes back as a clean explanation, and events
are read-only (only `list` / `read`).
"""

import json

import httpx
import pytest
import respx
from basecradle import BaseCradle

from basecradle_harness import (
    PlatformContext,
    PlatformError,
    WebhookEndpointsTool,
    WebhookEventsTool,
)

BC_URL = "https://basecradle.com"
FAKE_TOKEN = "bc_uat_KqI8zFxkQ0OZ8vYwT7mWcVtR3nSdLpEa"

NOVA_UUID = "019e7750-66ee-79c8-ad8a-bbb6ea7c2bcc"  # the agent
TIMELINE_UUID = "019e7750-66ee-7f53-829f-13a8a710b6da"  # the current timeline
OTHER_TIMELINE = "019e7760-1234-7abc-8def-0123456789ab"  # an explicit override

# Well-formed UUIDv7 endpoint and event (content) uuids.
EP_ONE = "019e7751-4a1b-7c2d-8e3f-1a2b3c4d5e6f"
EP_TWO = "019e7752-5b2c-7d3e-9f40-2b3c4d5e6f70"
EV_ONE = "019e7753-6c3d-7e4f-9051-3c4d5e6f7081"
EV_TWO = "019e7754-7d4e-7f50-a162-4d5e6f708192"

INGEST_URL = "https://basecradle.com/wh/ingest/wh_abc123def456"
ROTATED_URL = "https://basecradle.com/wh/ingest/wh_xyz789ghi012"


# --- wire payload builders ---------------------------------------------------


def endpoint(
    *,
    uuid=EP_ONE,
    description="Stripe deliveries",
    enabled=True,
    ingest_url=INGEST_URL,
    verification_enabled=False,
):
    """A webhook endpoint in subject form (the SDK's documented shape)."""
    return {
        "type": "webhook_endpoint",
        "created_at": "2026-06-04T00:00:00.000Z",
        "timeline": {"uuid": TIMELINE_UUID},
        "content": {
            "uuid": uuid,
            "description": description,
            "enabled": enabled,
            "ingest_url": ingest_url,
            "verification": {
                "enabled": verification_enabled,
                "signature_header": "X-BaseCradle-Signature",
                "verifier": "hmac_sha256_hex",
            },
        },
    }


def event(
    *,
    uuid=EV_ONE,
    content_type="application/json",
    headers=None,
    payload='{"event":"payment.succeeded"}',
    endpoint_uuid=EP_ONE,
):
    """A webhook event in subject form — one inbound delivery."""
    return {
        "type": "webhook_event",
        "created_at": "2026-06-05T00:00:00.000Z",
        "timeline": {"uuid": TIMELINE_UUID},
        "webhook_endpoint": {"uuid": endpoint_uuid},
        "content": {
            "uuid": uuid,
            "content_type": content_type,
            "headers": headers if headers is not None else {"User-Agent": "Stripe/1.0"},
            "payload": payload,
            "ingest_token_at_receipt": "wh_abc123def456",
        },
    }


def timeline_envelope(*, uuid=TIMELINE_UUID, name="Incident response"):
    """The subject-timeline envelope the SDK merges on get (needed before create)."""
    return {
        "timeline": {
            "uuid": uuid,
            "name": name,
            "locked": False,
            "created_at": "2026-06-01T00:00:00.000Z",
            "updated_at": "2026-06-02T00:00:00.000Z",
            "owner": {"uuid": NOVA_UUID, "handle": "nova", "name": "Nova Digital", "kind": "ai"},
            "participants": [
                {"uuid": NOVA_UUID, "handle": "nova", "name": "Nova Digital", "kind": "ai"}
            ],
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
def endpoints(client):
    """A WebhookEndpointsTool bound to Nova's current timeline through a real client."""
    t = WebhookEndpointsTool()
    t.bind(PlatformContext(client=client, timeline=TIMELINE_UUID))
    return t


@pytest.fixture
def events(client):
    """A WebhookEventsTool bound through a real client."""
    t = WebhookEventsTool()
    t.bind(PlatformContext(client=client, timeline=TIMELINE_UUID))
    return t


# --- endpoints: create -------------------------------------------------------


def test_create_makes_an_endpoint_and_reports_its_ingest_url(endpoints):
    captured = {}

    def capture(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(201, json={"webhook_endpoint": endpoint()})

    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/timelines/{TIMELINE_UUID}").mock(
            return_value=httpx.Response(200, json=timeline_envelope())
        )
        mock.post(f"{BC_URL}/timelines/{TIMELINE_UUID}/webhook_endpoints").mock(side_effect=capture)
        result = endpoints.run(action="create", description="Stripe deliveries")

    assert captured["body"] == {"webhook_endpoint": {"description": "Stripe deliveries"}}
    assert INGEST_URL in result  # the secret address is surfaced
    assert EP_ONE in result
    assert "verification is off" in result


def test_create_without_a_description_is_a_friendly_error(endpoints):
    # No routes mocked: a missing description must fail before any request goes out.
    assert "needs a 'description'" in endpoints.run(action="create")


def test_create_reports_when_signature_verification_is_on(endpoints):
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/timelines/{TIMELINE_UUID}").mock(
            return_value=httpx.Response(200, json=timeline_envelope())
        )
        mock.post(f"{BC_URL}/timelines/{TIMELINE_UUID}/webhook_endpoints").mock(
            return_value=httpx.Response(
                201, json={"webhook_endpoint": endpoint(verification_enabled=True)}
            )
        )
        result = endpoints.run(action="create", description="Signed sender")

    assert "verification is on" in result


def test_create_can_target_an_explicit_timeline(endpoints):
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/timelines/{OTHER_TIMELINE}").mock(
            return_value=httpx.Response(200, json=timeline_envelope(uuid=OTHER_TIMELINE))
        )
        post = mock.post(f"{BC_URL}/timelines/{OTHER_TIMELINE}/webhook_endpoints").mock(
            return_value=httpx.Response(201, json={"webhook_endpoint": endpoint()})
        )
        endpoints.run(action="create", description="elsewhere", timeline=OTHER_TIMELINE)

    assert post.called  # it created on the explicit timeline, not the bound one


def test_create_surfaces_a_refusal_as_a_clean_explanation(endpoints):
    detail = "You must be a viewer of this timeline to add a webhook endpoint."
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/timelines/{TIMELINE_UUID}").mock(
            return_value=httpx.Response(200, json=timeline_envelope())
        )
        mock.post(f"{BC_URL}/timelines/{TIMELINE_UUID}/webhook_endpoints").mock(
            return_value=httpx.Response(
                403, json=problem(status=403, code="not_a_viewer", detail=detail)
            )
        )
        result = endpoints.run(action="create", description="nope")

    assert "Couldn't create the endpoint" in result
    assert detail in result  # the server's own human explanation is relayed verbatim


# --- endpoints: list ---------------------------------------------------------


def test_list_renders_each_endpoint_with_uuid_state_and_ingest_url(endpoints):
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/webhook_endpoints").mock(
            return_value=httpx.Response(
                200,
                json={
                    "webhook_endpoints": [
                        endpoint(),
                        endpoint(uuid=EP_TWO, description="GitHub", enabled=False),
                    ],
                    "next_cursor": None,
                },
            )
        )
        result = endpoints.run(action="list")

    assert EP_ONE in result and EP_TWO in result
    assert INGEST_URL in result
    assert "enabled" in result and "disabled" in result  # both states rendered


def test_list_on_an_empty_timeline_says_so(endpoints):
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/webhook_endpoints").mock(
            return_value=httpx.Response(200, json={"webhook_endpoints": [], "next_cursor": None})
        )
        assert "No webhook endpoints" in endpoints.run(action="list")


def test_list_can_target_an_explicit_timeline(endpoints):
    with respx.mock(assert_all_called=True) as mock:
        route = mock.get(f"{BC_URL}/webhook_endpoints", params={"timeline": OTHER_TIMELINE}).mock(
            return_value=httpx.Response(200, json={"webhook_endpoints": [], "next_cursor": None})
        )
        endpoints.run(action="list", timeline=OTHER_TIMELINE)

    assert route.called  # it queried the other timeline, not the bound one


# --- endpoints: enable / disable / rotate ------------------------------------


def test_enable_turns_an_endpoint_on(endpoints):
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/webhook_endpoints/{EP_ONE}").mock(
            return_value=httpx.Response(200, json={"webhook_endpoint": endpoint(enabled=False)})
        )
        route = mock.post(f"{BC_URL}/webhook_endpoints/{EP_ONE}/enablement").mock(
            return_value=httpx.Response(200, json={"webhook_endpoint": endpoint(enabled=True)})
        )
        result = endpoints.run(action="enable", uuid=EP_ONE)

    assert route.called
    assert "Enabled webhook endpoint" in result


def test_disable_turns_an_endpoint_off_and_explains_the_soft_stop(endpoints):
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/webhook_endpoints/{EP_ONE}").mock(
            return_value=httpx.Response(200, json={"webhook_endpoint": endpoint(enabled=True)})
        )
        route = mock.delete(f"{BC_URL}/webhook_endpoints/{EP_ONE}/enablement").mock(
            return_value=httpx.Response(200, json={"webhook_endpoint": endpoint(enabled=False)})
        )
        result = endpoints.run(action="disable", uuid=EP_ONE)

    assert route.called
    assert "Disabled webhook endpoint" in result
    assert "410 Gone" in result
    assert "history is kept" in result


def test_rotate_reports_the_new_ingest_url_and_kills_the_old(endpoints):
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/webhook_endpoints/{EP_ONE}").mock(
            return_value=httpx.Response(200, json={"webhook_endpoint": endpoint()})
        )
        route = mock.post(f"{BC_URL}/webhook_endpoints/{EP_ONE}/rotation").mock(
            return_value=httpx.Response(
                200, json={"webhook_endpoint": endpoint(ingest_url=ROTATED_URL)}
            )
        )
        result = endpoints.run(action="rotate", uuid=EP_ONE)

    assert route.called
    assert ROTATED_URL in result  # the new URL is surfaced
    assert INGEST_URL not in result  # the old one is not echoed back
    assert "old ingest URL is dead" in result


def test_enable_without_a_uuid_is_a_friendly_error(endpoints):
    assert "needs the endpoint's uuid" in endpoints.run(action="enable")


def test_rotate_surfaces_a_refusal_as_a_clean_explanation(endpoints):
    detail = "This endpoint belongs to a timeline you no longer view."
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/webhook_endpoints/{EP_ONE}").mock(
            return_value=httpx.Response(
                403, json=problem(status=403, code="not_a_viewer", detail=detail)
            )
        )
        result = endpoints.run(action="rotate", uuid=EP_ONE)

    assert "Couldn't rotate the endpoint" in result
    assert detail in result


# --- endpoints: dispatch / binding -------------------------------------------


def test_endpoints_unknown_action_is_reported(endpoints):
    assert "unknown action" in endpoints.run(action="delete")


def test_endpoints_unbound_tool_raises_platform_error():
    with pytest.raises(PlatformError):
        WebhookEndpointsTool().run(action="list")


# --- events: list ------------------------------------------------------------


def test_list_events_renders_each_with_uuid_type_endpoint_and_preview(events):
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/webhook_events").mock(
            return_value=httpx.Response(
                200,
                json={
                    "webhook_events": [event(), event(uuid=EV_TWO, content_type="text/plain")],
                    "next_cursor": None,
                },
            )
        )
        result = events.run(action="list")

    assert EV_ONE in result and EV_TWO in result
    assert "application/json" in result and "text/plain" in result
    assert EP_ONE in result  # the originating endpoint is shown


def test_list_events_narrows_to_one_endpoint(events):
    with respx.mock(assert_all_called=True) as mock:
        route = mock.get(
            f"{BC_URL}/webhook_events", params={"timeline": TIMELINE_UUID, "endpoint": EP_ONE}
        ).mock(return_value=httpx.Response(200, json={"webhook_events": [], "next_cursor": None}))
        result = events.run(action="list", endpoint=EP_ONE)

    assert route.called  # the endpoint filter reached the wire
    assert "for that endpoint" in result  # the empty message names the narrowed scope


def test_list_events_on_an_empty_timeline_says_so(events):
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/webhook_events").mock(
            return_value=httpx.Response(200, json={"webhook_events": [], "next_cursor": None})
        )
        assert "No webhook events" in events.run(action="list")


def test_list_events_truncates_a_long_payload_in_the_preview(events):
    long = json.dumps({"data": "x" * 500})
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/webhook_events").mock(
            return_value=httpx.Response(
                200, json={"webhook_events": [event(payload=long)], "next_cursor": None}
            )
        )
        result = events.run(action="list")

    assert long not in result  # the full body is not dumped into a list
    assert "…" in result


# --- events: read ------------------------------------------------------------


def test_read_event_returns_headers_and_the_full_payload(events):
    payload = json.dumps({"event": "payment.succeeded", "amount": 4200})
    headers = {"User-Agent": "Stripe/1.0", "Content-Type": "application/json"}
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/webhook_events/{EV_ONE}").mock(
            return_value=httpx.Response(
                200, json={"webhook_event": event(payload=payload, headers=headers)}
            )
        )
        result = events.run(action="read", uuid=EV_ONE)

    assert payload in result  # the whole body, not a preview
    assert EV_ONE in result
    assert "User-Agent: Stripe/1.0" in result  # headers rendered
    assert EP_ONE in result  # the originating endpoint


def test_read_event_without_a_uuid_is_a_friendly_error(events):
    assert "needs the event's uuid" in events.run(action="read")


def test_list_events_surfaces_a_refusal_as_a_clean_explanation(events):
    detail = "You must be a viewer of this timeline to read its webhook events."
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/webhook_events").mock(
            return_value=httpx.Response(
                403, json=problem(status=403, code="not_a_viewer", detail=detail)
            )
        )
        result = events.run(action="list")

    assert "Couldn't list the events" in result
    assert detail in result


# --- events: dispatch / binding ----------------------------------------------


def test_events_unknown_action_is_reported(events):
    assert "unknown action" in events.run(action="create")


def test_events_unbound_tool_raises_platform_error():
    with pytest.raises(PlatformError):
        WebhookEventsTool().run(action="list")
