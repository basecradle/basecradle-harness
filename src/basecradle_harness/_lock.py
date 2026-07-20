"""The emergency stop, alone and guarded: lock a timeline, irreversibly.

Locking permanently freezes a timeline's content, and there is no unlock (reopening is
an operator-only console action). The capital's @jt test surfaced the failure this guards
against (finding B1): a model that meant to *list* or *delete* a timeline reflexively
grabbed `lock`, the cheapest matching action, and froze a room it never meant to touch.

The structural fix was to pull lock **out of the timelines tool entirely** and stand it up
as its own tool, so it can never be the accidental default of a benign management call. The
`timelines` tool is now pure benign management and reads; the one-way action lives here.

The gate itself is the shared `ConfirmedTimelineAction` convention (`_confirmed.py`): lock
runs only when `confirm` equals the **target timeline's uuid** — a deliberate, target-specific
yes a reflexive grab cannot fake and cannot aim at the wrong room — and a bare or mismatched
call gets a **preview** of what would be frozen plus the exact uuid to confirm with, touching
nothing destructive. Lock and its sibling `delete` (`_delete.py`) share this one gate; there
is no per-tool snowflake. (An earlier change had relaxed lock to a boolean `confirm=true`;
moving onto the shared base re-unifies the two and closes the wrong-target gap that boolean
left open.)

A `PlatformTool` via `ConfirmedTimelineAction`: it reaches the SDK client and current
timeline through the bound `PlatformContext`, exactly as the other platform tools do. I/O
discipline (safe-by-construction): the SDK is the only platform I/O, and nothing touches the
filesystem.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from basecradle_harness._confirmed import ConfirmedTimelineAction

if TYPE_CHECKING:
    from basecradle import Timeline


class LockTool(ConfirmedTimelineAction):
    """Permanently freeze a timeline — the irreversible emergency stop, behind uuid-confirm.

    A `ConfirmedTimelineAction`: the gate (confirm-by-uuid, preview-on-refuse, error relay)
    is inherited; this only declares the lock op and its wording. Until the hosting agent
    binds a `PlatformContext`, `run` reports it is not connected (via `PlatformError`).
    """

    name = "lock"
    verb = "lock"
    consequence = (
        "Locking permanently freezes the timeline's content and there is NO unlock "
        "(reopening is an operator-only action); it is NOT how you list, leave, or delete "
        "a timeline."
    )
    description = (
        "Permanently freeze a timeline's content — the IRREVERSIBLE emergency stop. There "
        "is NO unlock; reopening a locked timeline is an operator-only action. Because it "
        "cannot be undone, this tool acts only when you pass confirm=<the timeline's uuid> "
        "to deliberately target it — a call without the matching uuid is refused, changes "
        "nothing, and returns a preview of what would be frozen plus the uuid to confirm "
        "with. Locks the current timeline unless you pass a timeline uuid. This is NOT how "
        "you list, leave, or delete a timeline — to permanently DESTROY a timeline and its "
        "content, use the separate 'delete' tool; this one only freezes, forever. "
        "Platform REST: POST /timelines/{timeline_uuid}/lock — this tool calls that same "
        "endpoint, under the same confirm=uuid discipline (not a bypass); "
        "https://basecradle.com/docs/api.md has the full API."
    )

    def _perform(self, timeline: Timeline) -> None:
        timeline.lock()

    def succeeded(self, timeline: Timeline) -> str:
        return (
            f"Locked timeline {timeline.name!r} (uuid={timeline.uuid}). Its content is now "
            "frozen permanently — this is one-way; reopening it is an operator-only action."
        )
