"""The emergency stop, alone and guarded: lock a timeline, irreversibly.

Locking is the one **irreversible** action a Harness agent can take on the
platform — it permanently freezes a timeline's content, and there is no unlock
(reopening is an operator-only console action). The capital's @jt test surfaced the
failure this guards against (finding B1): a model that meant to *list* or *delete* a
timeline reflexively grabbed `lock`, the cheapest matching action, and froze a room
it never meant to touch.

The structural fix is to pull lock **out of the timelines tool entirely** and stand
it up as its own tool, so it can never be the accidental default of a benign
management call. The `timelines` tool is now pure benign management and reads
(create, read, list, add/remove a participant); the one-way action lives here, by
itself, behind a deliberate gate:

- `LockTool.run` refuses unless the model passes **`confirm=true`**. The bare call —
  the reflexive grab — changes nothing and comes back with an explanation of what
  lock is (and what it is *not* for). The model has to set `confirm` on purpose,
  which a reflexive tool-grab does not do.

This keeps Phase 1's containment intent — a lock must be deliberate — in a cleaner,
structurally-isolated form than the old in-tool confirm echo.

A `PlatformTool`: it reaches the SDK client and current timeline through the bound
`PlatformContext`, exactly as the other platform tools do. I/O discipline
(safe-by-construction): the SDK is the only platform I/O, and nothing touches the
filesystem.
"""

from __future__ import annotations

from basecradle import BaseCradleError

from basecradle_harness._platform import PlatformTool, explain


class LockTool(PlatformTool):
    """Permanently freeze a timeline — the irreversible emergency stop, behind `confirm`.

    A `PlatformTool`: the hosting agent binds the SDK client and current-timeline uuid
    before the loop runs. Until bound, `run` reports it is not connected (via
    `PlatformError`) rather than failing obscurely.
    """

    name = "lock"
    description = (
        "Permanently freeze a timeline's content — the IRREVERSIBLE emergency stop. There "
        "is NO unlock; reopening a locked timeline is an operator-only action. Because it "
        "cannot be undone, this tool does nothing unless you ALSO pass confirm=true to "
        "deliberately acknowledge that — a call without confirm=true is refused and changes "
        "nothing. Locks the current timeline unless you pass a timeline uuid. This is NOT "
        "how you list, leave, or delete a timeline — it only freezes one, forever."
    )
    parameters = {
        "type": "object",
        "properties": {
            "confirm": {
                "type": "boolean",
                "description": (
                    "Required to actually lock. Set it to true to deliberately confirm you "
                    "mean to permanently freeze this timeline. Without confirm=true the lock "
                    "is refused — this is the guard against an accidental, irreversible lock."
                ),
            },
            "timeline": {
                "type": "string",
                "description": (
                    "Optional timeline uuid to lock instead of the current one. Omit to lock "
                    "the timeline you are engaged on."
                ),
            },
        },
        "required": [],
    }

    def run(self, confirm: bool = False, timeline: str | None = None) -> str:
        """Lock `timeline` (or the current one) — but only when `confirm` is true."""
        target = timeline or self.context.timeline
        if not confirm:
            return (
                f"Refused to lock timeline {target}: locking is the IRREVERSIBLE emergency "
                "stop — it permanently freezes the timeline and there is no unlock "
                "(reopening is an operator-only action). It is NOT how you list, leave, or "
                "delete a timeline. If you truly mean to permanently freeze this timeline, "
                "call lock again with confirm=true."
            )
        try:
            frozen = self.context.client.timelines.get(target)
            frozen.lock()
        except BaseCradleError as error:
            return f"Couldn't lock the timeline: {explain(error)}"
        return (
            f"Locked timeline {frozen.name!r} (uuid={frozen.uuid}). Its content is now frozen "
            "permanently — this is one-way; reopening it is an operator-only action."
        )
