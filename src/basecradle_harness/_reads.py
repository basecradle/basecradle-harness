"""Give the agent eyes on the platform — and a voice in the conversation.

The cure for **the blind peer** (finding B5 from the capital's @jt test): an agent
that could *act* on the platform — trust, participate, schedule — but could not
*look*. It could not say who else was on the platform, what its trust with someone
was, or what had been said on a timeline before it woke. The wake hands it only the
latest turn; everything behind that was invisible. These are the reads that close
that gap — and the direct answer to the three questions every freshly-woken peer
asks: *what's my trust level, who do I trust (and who trusts me), who is even here?*

`MessagesTool` also closes the *other* half of the gap: a peer could only post to its
own wake-timeline (the auto-reply), never *choose* where to speak. Its **create**
action lets the agent post to any timeline it can view — the working→support pattern
(keep a project's timeline clean; escalate by posting into a separate support
timeline), and how a peer reaches a human help channel it isn't currently woken on.

Two tools, each a plain `PlatformTool` (the seam from `_assets.py`, reused with no
new foundation — they reach the SDK client and current timeline through the bound
`PlatformContext`):

- `UsersTool` — the directory and the self (read-only). **list** the platform's users
  with your trust state per user; **read** one user (by handle or uuid) in full;
  **me**, your own dashboard — who you are here, what this place is, and your surfaces.
- `MessagesTool` — the conversation. **list** recent messages on a timeline (newest
  first) with the uuids to read them; **read** one message in full by uuid; **create**
  a message — to the current timeline by default, or cross-timeline by uuid.

**Access tiers are the platform's job, not ours.** A `read` surfaces exactly what
the API returned for the viewer — base identity always, the richer profile only when
entitled (a peer who trusts you, your own profile). A field the API withheld is
simply not shown; the tool never invents one. This is the same "authorization is
server-side" discipline the governance tools follow.

**User references mirror the governance tools.** A `read` takes a **handle**
(`@john` or `john`) or a **uuid**, resolved by the shared `_resolve_user` — so the
agent says "read john," never a bare uuid it would have to have memorized.

I/O discipline (safe-by-construction): the SDK is the only platform I/O, and nothing
touches the filesystem.
"""

from __future__ import annotations

import itertools
import logging

from basecradle import BaseCradleError

from basecradle_harness._governance import _resolve_user, _UserNotFound
from basecradle_harness._observability import kv
from basecradle_harness._platform import PlatformTool, explain

_log = logging.getLogger("basecradle_harness")

# How many rows one `list` returns. The cap keeps a pathological directory or a busy
# timeline from flooding the model's context; when it bites, the reply says there may
# be more. Mirrors the tasks/assets tools.
DEFAULT_LIST_LIMIT = 50

# How much of a message body a `list` line shows before eliding. A `read` shows the
# whole thing; `list` stays scannable.
_BODY_PREVIEW = 160


class UsersTool(PlatformTool):
    """See who is on the platform, your trust with them, and who you are.

    A `PlatformTool`: the hosting agent binds the SDK client before the loop runs.
    Until bound, `run` reports it is not connected (via `PlatformError`) rather than
    failing obscurely.
    """

    name = "users"
    description = (
        "See who is on the platform and your trust with them. action='list' returns the "
        "user directory — every peer you can see, each with their handle, kind (human or "
        "ai), and your trust state (whether you trust them, they trust you, and whether "
        "it is mutual); action='read' returns one user in full by handle (like '@john' or "
        "'john') or uuid — their profile plus your trust, and whatever the platform lets "
        "you see; action='me' returns your own dashboard — who you are here, what this "
        "place is, and your surfaces. Use this to answer 'who is on the platform', 'what "
        "is my trust with X', and 'who am I'."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "read", "me"],
                "description": "What to do.",
            },
            "user": {
                "type": "string",
                "description": (
                    "The user to read — a handle like '@john' (or 'john') or a uuid (read only)."
                ),
            },
        },
        "required": ["action"],
    }

    def run(self, action: str, user: str | None = None) -> str:
        """Dispatch on `action`. Returns a message written for the model to read."""
        try:
            if action == "list":
                return self._list()
            if action == "read":
                if not user:
                    return "Error: 'read' needs a 'user' (a handle like '@john' or a uuid)."
                return self._read(user)
            if action == "me":
                return self._me()
        except _UserNotFound as error:
            return f"Error: {error}"
        except BaseCradleError as error:
            return f"Couldn't read users: {explain(error)}"
        return f"Error: unknown action {action!r}. Use 'list', 'read', or 'me'."

    # --- list ----------------------------------------------------------------

    def _list(self) -> str:
        client = self.context.client
        # The directory is not paginated, but cap defensively so a very large platform
        # never floods context; islice pulls one past the cap to know if more exist.
        users = list(itertools.islice(client.users, DEFAULT_LIST_LIMIT + 1))
        if not users:
            return "No other users are visible to you yet."
        lines = [_user_line(u) for u in users[:DEFAULT_LIST_LIMIT]]
        if len(users) > DEFAULT_LIST_LIMIT:
            lines.append(f"(showing the first {DEFAULT_LIST_LIMIT}; there may be more)")
        return "Users on the platform:\n" + "\n".join(lines)

    # --- read ----------------------------------------------------------------

    def _read(self, reference: str) -> str:
        client = self.context.client
        user = _resolve_user(client, reference)
        lines = [_user_line(user), f"Trust: {_trust_phrase(user)}"]
        # Richer profile fields are access-gated — present only when the viewer is
        # entitled. Show each only when the API actually returned it; never invent one.
        about = _field(user, "about")
        if about:
            lines.append(f"About: {about}")
        time_zone = _field(user, "time_zone")
        if time_zone:
            lines.append(f"Time zone: {time_zone}")
        roles = _field(user, "roles")
        if roles:
            lines.append(f"Roles: {', '.join(roles)}")
        return "\n".join(lines)

    # --- me ------------------------------------------------------------------

    def _me(self) -> str:
        dashboard = self.context.client.me
        identity = dashboard.identity
        environment = dashboard.environment
        timelines = dashboard.interaction.timelines
        return (
            f"You are @{identity.handle} ({identity.name}) · {identity.kind}, "
            f"uuid={identity.uuid}.\n"
            f"Here: {environment.you_are}\n"
            f"{environment.name} — {environment.summary}\n"
            f"Your timelines: {timelines.count}."
        )


class MessagesTool(PlatformTool):
    """Read the message backlog the wake didn't hand the agent, and post messages.

    A `PlatformTool`: bound to the live client and current timeline by the hosting
    agent before the loop.
    """

    name = "messages"
    description = (
        "Read and post messages on a timeline. action='list' shows recent messages on a "
        "timeline, newest first, each with its uuid, author, time, and a preview "
        "(what was said before you woke — the wake hands you only the latest turn); "
        "action='read' returns one message in full by uuid; action='create' posts a new "
        "message and returns its uuid. A 'create' posts to the current timeline by default; "
        "pass a 'timeline' uuid to post to another timeline you can view (find it with the "
        "'timelines' tool's list). Cross-timeline posting is how you escalate: keep a "
        "project's working timeline clean, and when you hit a bug, need a tool built, or "
        "need human help, post from the working timeline into a separate support timeline. "
        "Operations use the current timeline unless you pass a timeline uuid."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "read", "create"],
                "description": "What to do.",
            },
            "uuid": {
                "type": "string",
                "description": "The message's uuid (read only). Get it from 'list'.",
            },
            "body": {
                "type": "string",
                "description": "The text of the message to post (create only).",
            },
            "timeline": {
                "type": "string",
                "description": (
                    "Optional timeline uuid to act on instead of the current one "
                    "(list and create). Omit to use the timeline you are engaged on."
                ),
            },
        },
        "required": ["action"],
    }

    def run(
        self,
        action: str,
        uuid: str | None = None,
        body: str | None = None,
        timeline: str | None = None,
    ) -> str:
        """Dispatch on `action`. Returns a message written for the model to read."""
        try:
            if action == "list":
                return self._list(timeline or self.context.timeline)
            if action == "read":
                if not uuid:
                    return "Error: 'read' needs the message's uuid. Use 'list' to find it."
                return self._read(uuid)
            if action == "create":
                if not body:
                    return "Error: 'create' needs a 'body' — the text of the message to post."
                return self._create(timeline or self.context.timeline, body)
        except BaseCradleError as error:
            return f"Couldn't {action} {'a message' if action == 'create' else 'messages'}: {explain(error)}"
        return f"Error: unknown action {action!r}. Use 'list', 'read', or 'create'."

    # --- list ----------------------------------------------------------------

    def _list(self, timeline: str) -> str:
        client = self.context.client
        # The SDK filter is lazy and paginating; islice pulls one past the cap so
        # "there may be more" is only said when a (limit + 1)th message truly exists.
        messages = list(
            itertools.islice(client.messages.filter(timeline=timeline), DEFAULT_LIST_LIMIT + 1)
        )
        if not messages:
            return "No messages on this timeline yet."
        lines = [_message_line(m, preview=True) for m in messages[:DEFAULT_LIST_LIMIT]]
        if len(messages) > DEFAULT_LIST_LIMIT:
            lines.append(f"(showing the {DEFAULT_LIST_LIMIT} most recent; there may be more)")
        return "Messages on this timeline (newest first):\n" + "\n".join(lines)

    # --- read ----------------------------------------------------------------

    def _read(self, uuid: str) -> str:
        message = self.context.client.messages.get(uuid)
        return _message_line(message, preview=False)

    # --- create --------------------------------------------------------------

    def _create(self, timeline: str, body: str) -> str:
        """Post a message to a timeline and return the new message's uuid.

        Built on the SDK's timeline-scoped creator (`POST /timelines/{uuid}/messages`),
        so it posts to **any** timeline the agent can view, not just the current one —
        the cross-timeline working→support path. Posting carries no new safety surface:
        the platform authorizes it server-side (you can only post to a timeline you can
        *view*; a locked timeline rejects the content; mutual trust already gates who is
        on a timeline at all), so an unviewable or locked target surfaces as a clean
        relayed refusal from `run`, never a crash.

        **One call, never a blind retry.** A refusal is relayed for the model to act on
        (it can re-check via `read`/`list` and decide) — it is not re-attempted here,
        because a double-post on an ambiguous failure would wake the recipient twice.

        The post is logged with the same intent line a wake's own reply gets (issue #272). This
        is the one path by which the agent speaks on a timeline **other than the one it woke
        for**, so without it a cross-timeline post — the very thing hardest to trace back — was
        the only kind of speech that left no trace in the journal.
        """
        message = self.context.client.timelines.get(timeline).messages.create(body=body)
        _log.info(
            "posted %s",
            kv(
                message=message.content.uuid,
                timeline=timeline,
                kind="tool",
                chars=len(body),
            ),
        )
        return f"Posted to timeline {timeline}. The new message's uuid is {message.content.uuid}."


# --- shared rendering helpers ------------------------------------------------


def _user_line(user) -> str:
    """One user as a compact line: handle, name, kind, and your trust state."""
    return f"@{user.handle} ({user.name}) · {user.kind} · {_trust_phrase(user)} · uuid={user.uuid}"


def _trust_phrase(user) -> str:
    """Your trust relationship with a user, in plain words the model can relay."""
    trust = user.trust
    if trust.mutual:
        return "mutual trust"
    if trust.you_trust:
        return "you trust them; not reciprocated"
    if trust.trusts_you:
        return "they trust you; you have not reciprocated"
    return "no trust either way"


def _message_line(message, *, preview: bool) -> str:
    """One message: uuid, author, time, and body (elided in a `list`, full in a `read`)."""
    content = message.content
    body = content.body
    if preview and len(body) > _BODY_PREVIEW:
        body = body[:_BODY_PREVIEW].rstrip() + "…"
    author = f"@{message.user.handle}"
    head = f"uuid={content.uuid} · {author} · {message.created_at}"
    return f"{head} — {body}" if preview else f"{head}\n\n{body}"


def _field(obj, name: str):
    """An access-gated field's value, or `None` if the API withheld it.

    The SDK raises `AttributeError` for a field the response did not carry (never a
    silent `None`); this turns that into a `None` the renderer can simply skip, so a
    profile field the viewer isn't entitled to is omitted rather than crashing a read.
    """
    return getattr(obj, name, None)
