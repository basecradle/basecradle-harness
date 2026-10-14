"""Give the agent posts with a send time: create, list, read, and cancel tasks on a timeline.

The second tool that acts *on* the platform, and the proof that the seam from
`_assets.py` generalizes: this is a plain `PlatformTool` subclass with **no new
foundation** — it reaches the SDK client and current timeline through the bound
`PlatformContext`, exactly as `AssetsTool` does. A contributor adds the next
platform tranche by copying this same shape.

A task is **a message with a send time** (issue #674): an instruction, an
activation time, a status, and an author. Every viewer of its timeline can read it
from the moment it is created and is woken by it when it activates, so its
instructions are written to the room, each addressed by @handle to whoever should
act on it, the author included when the work is its own. Every rendering below
names the author. Four actions, the task equivalent of what a human peer does:

- **list** — what tasks are here, with the uuids needed to read them, each with
  its status and activation time.
- **read** — one task in full by uuid: its complete instructions, when it
  activates, and its current status.
- **create** — post a task from instructions the agent produced, with the
  time it should activate.
- **cancel** — withdraw one of the agent's own **pending** tasks by uuid before it
  activates (issue #675). It is what makes the dead man's switch buildable: schedule
  a task, then cancel-and-reschedule it at every check-in, so the last one fires only
  if the agent stops. The platform refuses a task someone else wrote (403) or one that
  is no longer pending (409), and both refusals come back as a plain `Error:` line.
  **Deliberately not a `confirm=` tool**, unlike lock and delete: those destroy other
  viewers' content, while a cancel removes only the caller's own unfired task.

Ops default to the **current** timeline (the one the agent is engaged on); an
explicit `timeline` uuid handles the rare cross-timeline case. (A `read` or a `cancel`
is by a task's own uuid, so it spans timelines you can view without an extra argument.)

The one thing a task needs that an asset doesn't is **when to activate**, and
`activate_at` is required. To read cleanly for a conversational agent, this tool
accepts it two ways and normalizes to a single absolute timestamp before handing
it to the SDK (see `_normalize_activate_at`):

- a **relative offset** — ``+<n><unit>``, unit one of ``s m h d w`` (seconds,
  minutes, hours, days, weeks): ``+90m``, ``+2h``, ``+1d``. Resolved against the
  current time *here, at call time*, so the agent never has to know the clock.
- an **absolute ISO-8601 timestamp** — ``2026-06-10T15:00:00Z`` (or with a
  ``+00:00`` offset, or none — a bare timestamp is read as UTC).

The relative form is the one to reach for in conversation ("remind me in two
hours" → ``+2h``); the absolute form covers a specific wall-clock time.

I/O discipline (safe-by-construction): the SDK is the only platform I/O, and
nothing touches the filesystem.
"""

from __future__ import annotations

import itertools
import re
from datetime import datetime, timedelta, timezone

from basecradle import BaseCradleError

from basecradle_harness._idempotency import TASK
from basecradle_harness._platform import PlatformTool, explain

# How many tasks one `list` returns. The cap keeps a pathological timeline from
# flooding the model's context; when it bites, the reply says there may be more.
DEFAULT_LIST_LIMIT = 50

# How much of a task's instructions a `list` line shows before eliding. A `read`
# shows the whole thing; `list` stays scannable.
_LIST_INSTRUCTIONS_PREVIEW = 120

# A relative activation offset: `+<n><unit>`, e.g. `+90m`, `+2h`, `+1d`.
_RELATIVE_RE = re.compile(r"^\+\s*(\d+)\s*([smhdw])$", re.IGNORECASE)
_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


class TasksTool(PlatformTool):
    """Create, list, read, and cancel scheduled tasks on the agent's current timeline.

    A `PlatformTool`: the hosting agent binds the SDK client and current-timeline
    uuid before the loop runs. Until bound, `run` reports it is not connected (via
    `PlatformError`) rather than failing obscurely.
    """

    name = "tasks"
    description = (
        "Post an instruction to the timeline at a chosen time. Every viewer can read it now "
        "and is woken by it at activate_at; write it as a message to the room, each "
        "instruction addressed by @handle (name yourself for your own follow-up). "
        "action='create' posts a task from the instructions you give and an activation time "
        "(activate_at); action='list' shows the tasks here with their uuids, status, "
        "activation time, and author; action='read' returns one task in full by uuid; "
        "action='cancel' withdraws one of your own pending tasks by uuid before it activates "
        "(cancel and re-create to reschedule). "
        "activate_at takes a relative offset like '+90m', '+2h', or '+1d' (seconds, "
        "minutes, hours, days, weeks — resolved from now), or an absolute ISO-8601 "
        "timestamp like '2026-06-10T15:00:00Z'. "
        "Operations use the current timeline unless you pass a timeline uuid. "
        "Platform REST: POST /timelines/{timeline_uuid}/tasks — this tool calls that same "
        "endpoint; https://basecradle.com/docs/api.md#tools-and-the-http-api has the full API."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "list", "read", "cancel"],
                "description": "What to do.",
            },
            "uuid": {
                "type": "string",
                "description": "The task's uuid (read and cancel). Get it from 'list'.",
            },
            "instructions": {
                "type": "string",
                "description": (
                    "What should happen when it activates, each instruction addressed to the "
                    "@handle(s) who should act — yourself included when the work is yours "
                    "(create only)."
                ),
            },
            "activate_at": {
                "type": "string",
                "description": (
                    "When the task activates (create only). A relative offset from now "
                    "like '+90m', '+2h', '+1d' (units: s, m, h, d, w), or an absolute "
                    "ISO-8601 timestamp like '2026-06-10T15:00:00Z'."
                ),
            },
            "timeline": {
                "type": "string",
                "description": (
                    "Optional timeline uuid to act on instead of the current one. "
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
        instructions: str | None = None,
        activate_at: str | None = None,
        timeline: str | None = None,
    ) -> str:
        """Dispatch on `action`. Returns a message written for the model to read."""
        target = timeline or self.context.timeline
        if action == "create":
            # Minted before the validation, never after: the ordinal is counted off the
            # transcript, which records this call either way (issue #297 — see `PlatformTool.key`).
            key = self.key(TASK)
            if not instructions or not activate_at:
                return "Error: 'create' needs both 'instructions' and 'activate_at'."
            return self._create(target, instructions, activate_at, key)
        if action == "list":
            return self._list(target)
        if action == "read":
            if not uuid:
                return "Error: 'read' needs the task's uuid. Use 'list' to find it."
            return self._read(uuid)
        if action == "cancel":
            if not uuid:
                return "Error: 'cancel' needs the task's uuid. Use 'list' to find it."
            return self._cancel(uuid)
        return f"Error: unknown action {action!r}. Use 'create', 'list', 'read', or 'cancel'."

    # --- create --------------------------------------------------------------

    def _create(
        self, timeline: str, instructions: str, activate_at: str, key: str | None = None
    ) -> str:
        """Schedule a task. `key` is the deterministic Idempotency-Key (issue #297).

        **A relative `activate_at` resolves against *now*, so a re-issue after a killed wake is not
        byte-identical** — `+90m` recovered forty minutes later means a different absolute time. The
        key still does its job, and its job is the one that matters: if the dead wake's create
        landed, the platform returns *that* task (the same key with a different body returns the
        first record), so the schedule the agent actually made is the one that stands and nothing is
        scheduled twice. Only if the create never landed does the re-issue schedule from the
        recovery's clock — later than intended, and visible in the task the tool reports back.
        """
        try:
            when = _normalize_activate_at(activate_at)
        except ValueError as error:
            return f"Error: {error}"
        client = self.context.client
        task = client.timelines.get(timeline).tasks.create(
            instructions=instructions, activate_at=when, idempotency_key=key
        )
        self.acted("task", task.content.uuid)  # scheduled work is a visible act (#293)
        return f"Scheduled a task. {_describe(task)}"

    # --- list ----------------------------------------------------------------

    def _list(self, timeline: str) -> str:
        client = self.context.client
        # Pull one past the cap so "there may be more" is only said when a
        # (DEFAULT_LIST_LIMIT + 1)th task actually exists — never on an exact 50.
        # The SDK filter is lazy and paginating, so islice fetches only what it needs.
        tasks = list(
            itertools.islice(client.tasks.filter(timeline=timeline), DEFAULT_LIST_LIMIT + 1)
        )
        if not tasks:
            return "No tasks on this timeline yet."
        lines = [_describe(task, preview=True) for task in tasks[:DEFAULT_LIST_LIMIT]]
        if len(tasks) > DEFAULT_LIST_LIMIT:
            lines.append(f"(showing the {DEFAULT_LIST_LIMIT} most recent; there may be more)")
        return "Tasks on this timeline (newest first):\n" + "\n".join(lines)

    # --- read ----------------------------------------------------------------

    def _read(self, uuid: str) -> str:
        client = self.context.client
        task = client.tasks.get(uuid)
        content = task.content
        return (
            f"uuid={content.uuid} · status={content.status} · "
            f"activate_at={content.activate_at} · by @{task.user.handle}\n\n{content.instructions}"
        )

    # --- cancel --------------------------------------------------------------

    def _cancel(self, uuid: str) -> str:
        """Withdraw a pending task. A platform refusal is relayed, never raised.

        The two refusals the model can act on — not the author (403), no longer pending
        (409) — arrive as the platform's own sentence, as does a uuid that names no task
        it can see (404). Not recorded with `acted`: a cancel puts nothing new on the
        timeline, the same as lock and delete.
        """
        client = self.context.client
        try:
            task = client.tasks.get(uuid)
            task.cancel()
        except BaseCradleError as error:
            return f"Error: couldn't cancel task {uuid}: {explain(error)}"
        return f"Cancelled the task. {_describe(task)}"


# --- shared rendering / normalization helpers --------------------------------


def _describe(task, *, preview: bool = False) -> str:
    """One task as a compact line: uuid, status, activation time, author, and instructions.

    With `preview`, instructions are elided to keep a `list` scannable; a `read`
    renders them in full elsewhere.
    """
    content = task.content
    instructions = content.instructions
    if preview and len(instructions) > _LIST_INSTRUCTIONS_PREVIEW:
        instructions = instructions[:_LIST_INSTRUCTIONS_PREVIEW].rstrip() + "…"
    return (
        f"uuid={content.uuid} · status={content.status} · "
        f"activate_at={content.activate_at} · by @{task.user.handle} — {instructions}"
    )


def _normalize_activate_at(value: str) -> str:
    """Resolve a relative offset or an ISO-8601 string to an absolute timestamp.

    A `+<n><unit>` offset is added to the current time; an absolute ISO-8601
    string is parsed and re-emitted in canonical form (a trailing ``Z`` is
    accepted, and a bare timestamp with no zone is read as UTC). Raises
    ``ValueError`` with a model-readable message on anything else.
    """
    text = value.strip()

    relative = _RELATIVE_RE.match(text)
    if relative:
        amount, unit = int(relative.group(1)), relative.group(2).lower()
        when = _utcnow() + timedelta(seconds=amount * _UNIT_SECONDS[unit])
        return when.isoformat()

    # Absolute ISO-8601. fromisoformat (3.10) doesn't take a trailing 'Z', so map
    # it to the equivalent offset; a naive timestamp is treated as UTC.
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(
            f"activate_at {value!r} is not a relative offset (like '+2h') "
            "or an ISO-8601 timestamp (like '2026-06-10T15:00:00Z')."
        ) from None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.isoformat()


def _utcnow() -> datetime:
    """Current UTC time. A seam tests patch to make relative offsets deterministic."""
    return datetime.now(timezone.utc)
