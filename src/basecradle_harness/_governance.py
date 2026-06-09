"""Give the agent owner-level governance: its own timelines, and its trust edges.

The third Phase-2 tranche, and more proof the platform seam generalizes: two more
`PlatformTool` subclasses with **no new foundation** — they reach the SDK client
and current timeline through the bound `PlatformContext`, exactly as `AssetsTool`
and `TasksTool` do. Governance spans two resource domains, so it ships as two
focused tools rather than one muddy catch-all — one resource per tool, the shape
assets and tasks established:

- `TimelinesTool` — a peer running its own rooms: **create** a timeline it owns,
  **add** / **remove** a participant, and **lock** a timeline (the emergency stop).
- `TrustTool` — a peer managing its own outgoing trust edges: **grant** or
  **revoke** trust toward another user. Trust is the consent currency that gates
  sharing a timeline (adding a participant needs *mutual* trust), so the two tools
  work in concert: trust someone, then add them.

**Authorization is the platform's job, not ours.** Adding a participant requires
ownership, mutual trust with every existing viewer, and headroom; removing one
requires ownership too. Locking is the emergency stop — open to any viewer, by
design (anyone in the room can pull it). All of that is enforced server-side. A
denied action comes back as a typed `BaseCradleError` whose `detail` is written
for a human — so these tools **catch it and surface that explanation** rather than
letting the agent flail on a raw traceback. Trusting is self-scoped (your own
outgoing edge) and always permitted; the platform silently ignores trusting
yourself.

**Lock is intentionally one-way.** There is no unlock in the platform API or the
SDK, by design: unlocking a locked timeline is an operator-only console action.
So `TimelinesTool` locks only — it never pretends to offer an unlock, and says so.

**User resolution.** A conversational agent says "trust @origin" or "add nova to
this timeline," not a uuid. Both tools accept a user reference as either a
**handle** (with or without a leading `@`, resolved by scanning `bc.users`) or a
**uuid** (resolved directly via `bc.users.get`). See `_resolve_user`.

I/O discipline (safe-by-construction): the SDK is the only platform I/O, and
nothing touches the filesystem.
"""

from __future__ import annotations

import re

from basecradle import BaseCradleError, User

from basecradle_harness._platform import PlatformTool

# A user reference that is a uuid goes straight to `bc.users.get`; anything else is
# treated as a handle and resolved by scanning the directory. Standard UUID shape
# (the platform issues UUIDv7, which matches this just the same).
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
)


class _UserNotFound(Exception):
    """No user matched a handle reference — raised by `_resolve_user`, caught in `run`."""


class TimelinesTool(PlatformTool):
    """Create, lock, and manage participants on timelines the agent owns.

    A `PlatformTool`: the hosting agent binds the SDK client and current-timeline
    uuid before the loop runs. Until bound, `run` reports it is not connected (via
    `PlatformError`) rather than failing obscurely.
    """

    name = "timelines"
    description = (
        "Run your own timelines. action='create' makes a new timeline you own from a "
        "name; action='lock' permanently freezes a timeline's content (the emergency "
        "stop — this is ONE-WAY, there is no unlock; reopening a locked timeline is an "
        "operator-only action); action='add_participant' adds a user to a timeline you "
        "own; action='remove_participant' removes one. A 'user' is a handle like "
        "'@nova' (or 'nova') or a uuid. Locking is the emergency stop, open to any "
        "viewer; adding or removing a participant needs you to own the timeline, and "
        "adding someone also needs mutual trust with them — if the platform refuses, "
        "you get the reason back. Timeline-scoped actions use the current timeline "
        "unless you pass a timeline uuid."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "lock", "add_participant", "remove_participant"],
                "description": "What to do.",
            },
            "name": {
                "type": "string",
                "description": "The name for the new timeline (create only).",
            },
            "user": {
                "type": "string",
                "description": (
                    "The user to add or remove — a handle like '@nova' (or 'nova') or a "
                    "uuid (add_participant / remove_participant only)."
                ),
            },
            "timeline": {
                "type": "string",
                "description": (
                    "Optional timeline uuid to act on instead of the current one "
                    "(lock / add_participant / remove_participant). Omit to use the "
                    "timeline you are engaged on."
                ),
            },
        },
        "required": ["action"],
    }

    # Human-readable verb per action, for a clean message when the platform refuses.
    _PHRASE = {
        "create": "create the timeline",
        "lock": "lock the timeline",
        "add_participant": "add the participant",
        "remove_participant": "remove the participant",
    }

    def run(
        self,
        action: str,
        name: str | None = None,
        user: str | None = None,
        timeline: str | None = None,
    ) -> str:
        """Dispatch on `action`. Returns a message written for the model to read."""
        if action not in self._PHRASE:
            return (
                f"Error: unknown action {action!r}. Use 'create', 'lock', "
                "'add_participant', or 'remove_participant'."
            )
        try:
            if action == "create":
                if not name:
                    return "Error: 'create' needs a 'name' for the timeline."
                return self._create(name)
            target = timeline or self.context.timeline
            if action == "lock":
                return self._lock(target)
            # add_participant / remove_participant
            if not user:
                return f"Error: '{action}' needs a 'user' (a handle like '@nova' or a uuid)."
            resolved = _resolve_user(self.context.client, user)
            if action == "add_participant":
                return self._add_participant(target, resolved)
            return self._remove_participant(target, resolved)
        except _UserNotFound as error:
            return f"Error: {error}"
        except BaseCradleError as error:
            return f"Couldn't {self._PHRASE[action]}: {_explain(error)}"

    # --- actions -------------------------------------------------------------

    def _create(self, name: str) -> str:
        timeline = self.context.client.timelines.create(name=name)
        return f"Created timeline {timeline.name!r} (uuid={timeline.uuid}). You own it."

    def _lock(self, target: str) -> str:
        timeline = self.context.client.timelines.get(target)
        timeline.lock()
        return (
            f"Locked timeline {timeline.name!r} (uuid={timeline.uuid}). Its content is now "
            "frozen permanently — this is one-way; reopening it is an operator-only action."
        )

    def _add_participant(self, target: str, user: User) -> str:
        timeline = self.context.client.timelines.get(target)
        timeline.add_participant(user)
        return f"Added @{user.handle} to timeline {timeline.name!r} (uuid={timeline.uuid})."

    def _remove_participant(self, target: str, user: User) -> str:
        timeline = self.context.client.timelines.get(target)
        timeline.remove_participant(user)
        return f"Removed @{user.handle} from timeline {timeline.name!r} (uuid={timeline.uuid})."


class TrustTool(PlatformTool):
    """Grant or revoke the agent's own outgoing trust toward another user.

    Trust is self-scoped — you only ever change your *own* outgoing edge, so it is
    always permitted (the platform silently ignores trusting yourself). Mutual
    trust, the thing that lets you share a timeline, also needs the other user to
    trust you back; this tool reports whether that is the case after a grant.

    A `PlatformTool`: bound to the live client by the hosting agent before the loop.
    """

    name = "trust"
    description = (
        "Manage your trust toward another user — the consent that gates sharing a "
        "timeline. action='grant' adds your trust edge toward a user; action='revoke' "
        "removes it. A 'user' is a handle like '@origin' (or 'origin') or a uuid. "
        "Trust is one-directional: granting means YOU trust them; sharing a timeline "
        "needs mutual trust (them trusting you back too)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["grant", "revoke"],
                "description": "Whether to grant or revoke your trust.",
            },
            "user": {
                "type": "string",
                "description": "The user to trust or untrust — a handle like '@origin' or a uuid.",
            },
        },
        "required": ["action", "user"],
    }

    _PHRASE = {"grant": "grant trust", "revoke": "revoke trust"}

    def run(self, action: str, user: str | None = None) -> str:
        """Dispatch on `action`. Returns a message written for the model to read."""
        if action not in self._PHRASE:
            return f"Error: unknown action {action!r}. Use 'grant' or 'revoke'."
        if not user:
            return f"Error: '{action}' needs a 'user' (a handle like '@origin' or a uuid)."
        try:
            resolved = _resolve_user(self.context.client, user)
            if action == "grant":
                return self._grant(resolved)
            return self._revoke(resolved)
        except _UserNotFound as error:
            return f"Error: {error}"
        except BaseCradleError as error:
            return f"Couldn't {self._PHRASE[action]}: {_explain(error)}"

    # --- actions -------------------------------------------------------------

    def _grant(self, user: User) -> str:
        user.grant_trust()
        if user.trust.mutual:
            return f"You now trust @{user.handle}, and they trust you — trust is mutual."
        return (
            f"You now trust @{user.handle}. Trust is not yet mutual: they have not trusted "
            "you back, so you cannot share a timeline until they do."
        )

    def _revoke(self, user: User) -> str:
        user.revoke_trust()
        return (
            f"You no longer trust @{user.handle}. Anyone already sharing a timeline with you "
            "stays — the trust gate only runs when a participation is created."
        )


# --- shared helpers ----------------------------------------------------------


def _resolve_user(client, reference: str) -> User:
    """Resolve a user reference (a handle or a uuid) to a live `User`.

    A uuid goes straight to `bc.users.get`. Anything else is treated as a handle:
    a leading `@` is stripped and the directory (`bc.users`) is scanned for a
    case-insensitive match. Raises `_UserNotFound` if no directory user matches the
    handle — a clean, model-readable miss rather than a confusing API error.
    """
    ref = reference.strip()
    if _UUID_RE.match(ref):
        return client.users.get(ref)
    handle = ref.lstrip("@").lower()
    for user in client.users:
        if user.handle.lower() == handle:
            return user
    raise _UserNotFound(
        f"No user with handle '@{handle}' is visible to you. Check the handle, or use a uuid."
    )


def _explain(error: BaseCradleError) -> str:
    """The most human-readable string a platform error carries.

    API errors are RFC 9457 problem documents: `detail` is the human sentence, with
    `title` and the raw message as fallbacks. This is what makes a refused action an
    explanation the agent can relay, not a traceback.
    """
    return error.detail or error.title or str(error)
