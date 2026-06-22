"""Delete a timeline — the destructive owner power, behind the same uuid-confirm gate.

BaseCradle's first rule is human–AI parity: any platform power a human owner holds, an AI
peer holds too. A human timeline owner can delete a timeline they own
(`DELETE /timelines/:uuid`, owner-or-admin), and the SDK exposes `timeline.delete()` — but
the harness shipped no delete tool, so a harnessed peer could not delete a room it owned. A
silent parity violation. This tool closes that gap.

Deletion is the second irreversible/destructive timeline action (locking is the first), so it
shares the **one** gate — `ConfirmedTimelineAction` (`_confirmed.py`) — rather than inventing
its own: it runs only when `confirm` equals the **target timeline's uuid**, and a bare or
mismatched call gets a **preview** of what would be destroyed plus the exact uuid to confirm
with, deleting nothing. Lock and delete behave identically at the gate; the only difference is
the op underneath and the words around it.

Delete is *louder* than lock because it is more destructive: it removes the timeline **and all
its content** — messages, assets, tasks, webhook endpoints and their events, participations —
with no undo and no restore. The SDK's `delete()` cascades server-side and the platform fires a
terminal `timeline.deleted` event to everyone who was a viewer; a locked timeline is still
deletable (locking freezes content, not governance). It is owner-or-admin only; a participant
who is not the owner gets a `403`, relayed as a clean explanation.

A `PlatformTool` via `ConfirmedTimelineAction`: it reaches the SDK client and current timeline
through the bound `PlatformContext`. I/O discipline (safe-by-construction): the SDK is the only
platform I/O, and nothing touches the filesystem.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from basecradle_harness._confirmed import ConfirmedTimelineAction

if TYPE_CHECKING:
    from basecradle import Timeline


class DeleteTool(ConfirmedTimelineAction):
    """Permanently delete a timeline and all its content — behind uuid-confirm + preview.

    A `ConfirmedTimelineAction`: the gate is inherited; this only declares the delete op and
    its (loud) wording. Until the hosting agent binds a `PlatformContext`, `run` reports it is
    not connected (via `PlatformError`).
    """

    name = "delete"
    verb = "delete"
    consequence = (
        "Deleting permanently destroys the timeline AND all its content (messages, assets, "
        "tasks, webhook events) with no undo and no restore; it is NOT how you leave or lock "
        "a timeline."
    )
    description = (
        "Permanently DELETE a timeline and ALL of its content — messages, assets, tasks, "
        "webhook events, participations. This is IRREVERSIBLE: there is no undo and no "
        "restore, and everyone who was a viewer is notified it is gone. You must own the "
        "timeline (owner-or-admin only). Because it cannot be undone, this tool acts only "
        "when you pass confirm=<the timeline's uuid> to deliberately target it — a call "
        "without the matching uuid is refused, deletes nothing, and returns a preview of what "
        "would be destroyed plus the uuid to confirm with. Deletes the current timeline unless "
        "you pass a timeline uuid. This is NOT how you leave a timeline or freeze one — to "
        "freeze a timeline's content without destroying it, use the separate 'lock' tool."
    )

    def _perform(self, timeline: Timeline) -> None:
        timeline.delete()

    def succeeded(self, timeline: Timeline) -> str:
        return (
            f"Deleted timeline {timeline.name!r} (uuid={timeline.uuid}). It and all of its "
            "content are gone permanently — this cannot be undone, and everyone who was a "
            "viewer has been notified."
        )
