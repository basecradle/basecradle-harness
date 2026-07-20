"""Give the agent inbound webhooks: manage endpoints, and read what arrives.

The final SDK tranche, and one more proof the platform seam generalizes: two more
`PlatformTool` subclasses with **no new foundation** — they reach the SDK client
and current timeline through the bound `PlatformContext`, exactly as `AssetsTool`,
`TasksTool`, and the governance tools do.

A **webhook endpoint** is an inbound URL on a timeline: an external service or
script `POST`s to its **ingest URL**, and each delivery is recorded as a **webhook
event** on the timeline. Endpoints are managed (created, enabled, rotated); events
are read-only — they exist only because something was delivered. That natural split
is the SDK's own (`webhook_endpoints` vs. `webhook_events`), so this ships as two
focused tools, one resource each — the shape governance set:

- `WebhookEndpointsTool` — wire a timeline up to receive activity: **create** an
  endpoint (and report its ingest URL — the thing you hand an external service),
  **list** the endpoints here, **enable** / **disable** one (the soft stop), and
  **rotate** one's ingest URL (the move when a URL leaks — the old one dies at once).
- `WebhookEventsTool` — inspect what is arriving: **list** the inbound deliveries on
  a timeline (optionally narrowed to one endpoint), and **read** one in full by uuid
  (its headers and raw payload).

Ops default to the **current** timeline (the one the agent is engaged on); an
explicit `timeline` uuid handles the rare cross-timeline case. (A `read`, and the
endpoint verbs, key off a resource's own uuid, so they span timelines you can view
without an extra argument.)

**The ingest URL is the secret.** It is the only credential an inbound sender needs,
so `create` and `rotate` surface it plainly, and `rotate` is the response to a leak.
**Out of scope by design:** an endpoint's *signature secret* (its
`verification_credential`) — that is a write-only owner action on the endpoint's own
page, and the SDK does not expose it. These tools manage endpoint lifecycle and read
events; they never pretend to set or rotate a signing secret, and the endpoint line
reports only *whether* signature verification is on.

**Authorization is the platform's job, not ours.** Creating, enabling, disabling,
and rotating are timeline-scoped and enforced server-side; a refused action comes
back as a typed `BaseCradleError` whose `detail` is written for a human, so these
tools **catch it and surface that explanation** rather than letting the agent flail
on a raw traceback.

I/O discipline (safe-by-construction): the SDK is the only platform I/O, and
nothing touches the filesystem.
"""

from __future__ import annotations

import itertools

from basecradle import BaseCradleError

from basecradle_harness._idempotency import WEBHOOK_ENDPOINT
from basecradle_harness._platform import PlatformTool, explain

# How many endpoints/events one `list` returns. The cap keeps a pathological
# timeline from flooding the model's context; when it bites, the reply says there
# may be more.
DEFAULT_LIST_LIMIT = 50

# How much of an event's payload a `list` line shows before eliding. A `read` shows
# the whole thing; `list` stays scannable.
_PAYLOAD_PREVIEW = 120


class WebhookEndpointsTool(PlatformTool):
    """Create, list, enable, disable, and rotate inbound webhook endpoints.

    A `PlatformTool`: the hosting agent binds the SDK client and current-timeline
    uuid before the loop runs. Until bound, `run` reports it is not connected (via
    `PlatformError`) rather than failing obscurely.
    """

    name = "webhook_endpoints"
    description = (
        "Manage inbound webhook endpoints — URLs external services POST to so their "
        "activity lands on the timeline. action='create' makes an endpoint from a "
        "description and reports its ingest URL (the secret URL you hand the external "
        "service); action='list' shows the endpoints here with their uuids, ingest "
        "URL, and enabled state; action='enable' / action='disable' turns one on or "
        "off (disable is a reversible soft stop — deliveries get 410 Gone, history is "
        "kept); action='rotate' regenerates an endpoint's ingest URL, killing the old "
        "one immediately (do this if a URL leaks). enable/disable/rotate take the "
        "endpoint's uuid (get it from 'list'). Operations use the current timeline "
        "unless you pass a timeline uuid. (Setting an endpoint's signature secret is "
        "not available here — that's an owner action on the endpoint's own page.) "
        "Platform REST: POST /timelines/{timeline_uuid}/webhook_endpoints — this tool calls that "
        "same endpoint; https://basecradle.com/docs/api.md#tools-and-the-http-api has the full API."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "list", "enable", "disable", "rotate"],
                "description": "What to do.",
            },
            "description": {
                "type": "string",
                "description": (
                    "A human label for the new endpoint — what it is for (create only)."
                ),
            },
            "uuid": {
                "type": "string",
                "description": (
                    "The endpoint's uuid (enable / disable / rotate only). Get it from 'list'."
                ),
            },
            "timeline": {
                "type": "string",
                "description": (
                    "Optional timeline uuid to act on instead of the current one "
                    "(create / list). Omit to use the timeline you are engaged on."
                ),
            },
        },
        "required": ["action"],
    }

    # Human-readable verb per action, for a clean message when the platform refuses.
    _PHRASE = {
        "create": "create the endpoint",
        "list": "list the endpoints",
        "enable": "enable the endpoint",
        "disable": "disable the endpoint",
        "rotate": "rotate the endpoint",
    }

    def run(
        self,
        action: str,
        description: str | None = None,
        uuid: str | None = None,
        timeline: str | None = None,
    ) -> str:
        """Dispatch on `action`. Returns a message written for the model to read."""
        if action not in self._PHRASE:
            return (
                f"Error: unknown action {action!r}. Use 'create', 'list', 'enable', "
                "'disable', or 'rotate'."
            )
        try:
            if action == "create":
                # Minted before the validation, never after: the ordinal is counted off the
                # transcript, which records this call either way (#297 — see `PlatformTool.key`).
                key = self.key(WEBHOOK_ENDPOINT)
                if not description:
                    return "Error: 'create' needs a 'description' for the endpoint."
                return self._create(timeline or self.context.timeline, description, key)
            if action == "list":
                return self._list(timeline or self.context.timeline)
            # enable / disable / rotate — keyed off the endpoint's own uuid.
            if not uuid:
                return f"Error: '{action}' needs the endpoint's uuid. Use 'list' to find it."
            if action == "enable":
                return self._set_enabled(uuid, enabled=True)
            if action == "disable":
                return self._set_enabled(uuid, enabled=False)
            return self._rotate(uuid)
        except BaseCradleError as error:
            return f"Couldn't {self._PHRASE[action]}: {explain(error)}"

    # --- actions -------------------------------------------------------------

    def _create(self, timeline: str, description: str, key: str | None = None) -> str:
        """Create an ingest endpoint. `key` is the deterministic Idempotency-Key (issue #297).

        The one create whose duplicate would be quietly expensive rather than merely noisy: a second
        endpoint means a second secret ingest URL, and the sender only ever holds the first.
        """
        endpoint = self.context.client.timelines.get(timeline).webhook_endpoints.create(
            description=description, idempotency_key=key
        )
        content = endpoint.content
        return (
            f"Created webhook endpoint (uuid={content.uuid}). Its ingest URL — the secret "
            f"address an external service POSTs to — is:\n\n{content.ingest_url}\n\n"
            "Hand that to the sender; rotate it if it ever leaks. Signature verification "
            f"is {'on' if content.verification.enabled else 'off'}."
        )

    def _list(self, timeline: str) -> str:
        # Pull one past the cap so "there may be more" is only said when a
        # (DEFAULT_LIST_LIMIT + 1)th endpoint actually exists. The SDK filter is lazy
        # and paginating, so islice fetches only what it needs.
        endpoints = list(
            itertools.islice(
                self.context.client.webhook_endpoints.filter(timeline=timeline),
                DEFAULT_LIST_LIMIT + 1,
            )
        )
        if not endpoints:
            return "No webhook endpoints on this timeline yet."
        lines = [_describe_endpoint(e) for e in endpoints[:DEFAULT_LIST_LIMIT]]
        if len(endpoints) > DEFAULT_LIST_LIMIT:
            lines.append(f"(showing the {DEFAULT_LIST_LIMIT} most recent; there may be more)")
        return "Webhook endpoints on this timeline (newest first):\n" + "\n".join(lines)

    def _set_enabled(self, uuid: str, *, enabled: bool) -> str:
        endpoint = self.context.client.webhook_endpoints.get(uuid)
        if enabled:
            endpoint.enable()
            return f"Enabled webhook endpoint {uuid}. Inbound deliveries are accepted again."
        endpoint.disable()
        return (
            f"Disabled webhook endpoint {uuid}. Inbound deliveries are refused (410 Gone) "
            "until you enable it again; its event history is kept."
        )

    def _rotate(self, uuid: str) -> str:
        endpoint = self.context.client.webhook_endpoints.get(uuid)
        endpoint.rotate()
        return (
            f"Rotated webhook endpoint {uuid}. The old ingest URL is dead; its new ingest "
            f"URL is:\n\n{endpoint.content.ingest_url}\n\nHand the new URL to the sender."
        )


class WebhookEventsTool(PlatformTool):
    """List and read the inbound deliveries recorded on a timeline.

    Events are read-only — they exist only because an external sender delivered to
    an endpoint's ingest URL — so this tool only ever reads. A `PlatformTool`: bound
    to the live client by the hosting agent before the loop.
    """

    name = "webhook_events"
    description = (
        "Inspect inbound webhook deliveries — what external services have POSTed to "
        "this timeline's endpoints. action='list' shows the events here with their "
        "uuids, time, content type, and a payload preview (optionally narrowed to one "
        "endpoint via 'endpoint'); action='read' returns one event in full by uuid — "
        "its headers and the raw payload exactly as delivered. Events are read-only. "
        "Operations use the current timeline unless you pass a timeline uuid. "
        "Platform REST: GET /webhook_events — this tool calls that same endpoint; "
        "https://basecradle.com/docs/api.md#tools-and-the-http-api has the full API."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "read"],
                "description": "What to do.",
            },
            "uuid": {
                "type": "string",
                "description": "The event's uuid (read only). Get it from 'list'.",
            },
            "endpoint": {
                "type": "string",
                "description": (
                    "Optional endpoint uuid to narrow 'list' to one endpoint's deliveries. "
                    "Omit to list every endpoint's events on the timeline."
                ),
            },
            "timeline": {
                "type": "string",
                "description": (
                    "Optional timeline uuid to act on instead of the current one (list only). "
                    "Omit to use the timeline you are engaged on."
                ),
            },
        },
        "required": ["action"],
    }

    def run(
        self,
        action: str,
        uuid: str | None = None,
        endpoint: str | None = None,
        timeline: str | None = None,
    ) -> str:
        """Dispatch on `action`. Returns a message written for the model to read."""
        try:
            if action == "list":
                return self._list(timeline or self.context.timeline, endpoint)
            if action == "read":
                if not uuid:
                    return "Error: 'read' needs the event's uuid. Use 'list' to find it."
                return self._read(uuid)
        except BaseCradleError as error:
            verb = "list the events" if action == "list" else "read the event"
            return f"Couldn't {verb}: {explain(error)}"
        return f"Error: unknown action {action!r}. Use 'list' or 'read'."

    # --- actions -------------------------------------------------------------

    def _list(self, timeline: str, endpoint: str | None) -> str:
        # `filter` ignores a None endpoint, so this narrows only when one is given.
        events = list(
            itertools.islice(
                self.context.client.webhook_events.filter(timeline=timeline, endpoint=endpoint),
                DEFAULT_LIST_LIMIT + 1,
            )
        )
        if not events:
            scope = " for that endpoint" if endpoint else ""
            return f"No webhook events on this timeline{scope} yet."
        lines = [_describe_event(e) for e in events[:DEFAULT_LIST_LIMIT]]
        if len(events) > DEFAULT_LIST_LIMIT:
            lines.append(f"(showing the {DEFAULT_LIST_LIMIT} most recent; there may be more)")
        return "Webhook events on this timeline (newest first):\n" + "\n".join(lines)

    def _read(self, uuid: str) -> str:
        event = self.context.client.webhook_events.get(uuid)
        content = event.content
        headers = "\n".join(f"  {key}: {value}" for key, value in content.headers.items())
        return (
            f"uuid={content.uuid} · received={event.created_at} · "
            f"content_type={content.content_type} · "
            f"endpoint={event.webhook_endpoint.uuid}\n\n"
            f"Headers:\n{headers or '  (none)'}\n\nPayload:\n{content.payload}"
        )


# --- shared rendering / error helpers ----------------------------------------


def _describe_endpoint(endpoint) -> str:
    """One endpoint as a compact line: uuid, enabled state, ingest URL, verification."""
    content = endpoint.content
    state = "enabled" if content.enabled else "disabled"
    verification = "signed" if content.verification.enabled else "unsigned"
    return (
        f"uuid={content.uuid} · {state} · {verification} · "
        f"ingest_url={content.ingest_url} — {content.description}"
    )


def _describe_event(event) -> str:
    """One event as a compact line: uuid, time, content type, endpoint, payload preview."""
    content = event.content
    payload = content.payload
    if len(payload) > _PAYLOAD_PREVIEW:
        payload = payload[:_PAYLOAD_PREVIEW].rstrip() + "…"
    return (
        f"uuid={content.uuid} · received={event.created_at} · "
        f"content_type={content.content_type} · endpoint={event.webhook_endpoint.uuid} "
        f"— {payload}"
    )
