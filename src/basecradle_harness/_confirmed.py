"""The one convention for irreversible timeline actions: confirm-by-uuid, preview-on-refuse.

Locking and deleting are the only two **irreversible/destructive** actions a Harness
agent can take on a timeline — locking freezes it forever (no unlock), deleting destroys
it and all its content with no restore. Both share the same danger and so must share the
same gate, in one place, so they can never drift into per-tool snowflakes.

`ConfirmedTimelineAction` is that one place. It implements the gate once; a subclass only
declares *what* the irreversible op is (`lock` vs `delete`) and the words around it:

- **Confirm by uuid, not a boolean.** The `confirm` argument must equal the **target
  timeline's uuid** — a deliberate, target-specific yes that a reflexive tool-grab cannot
  fake, and that cannot be aimed at the wrong room (an earlier boolean `confirm=true`
  could). A bare or mismatched confirm performs **no** destructive call.
- **Preview-on-refuse.** A refusal is not a dead end: the base does one **benign GET** to
  fetch the timeline's name and item count, then returns a refusal that *names what would
  be affected* and hands back the exact `confirm=<uuid>` to re-call with. The model is told
  precisely what it nearly did and how to proceed on purpose.
- **Confirm == target → act.** Run the subclass's SDK op, relay any `BaseCradleError` as a
  clean explanation (never a raw traceback), and return the subclass's success message.

A subclass supplies four small hooks: `name`/`description` (a `Tool` already requires
these), `verb` (the word in every message — "lock"/"delete"), `consequence` (the one-line
warning the refusal carries), `_perform` (the SDK call), and `succeeded` (the success
sentence). Everything else — the parameter schema, the gate, the preview, the error relay —
is inherited and identical across actions.

A `PlatformTool`: it reaches the SDK client and current timeline through the bound
`PlatformContext`, exactly as the other platform tools do. I/O discipline
(safe-by-construction): the SDK is the only platform I/O, and nothing touches the filesystem.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from basecradle import BaseCradleError

from basecradle_harness._platform import PlatformTool, explain

if TYPE_CHECKING:
    from basecradle import Timeline


class ConfirmedTimelineAction(PlatformTool):
    """Base for an irreversible timeline action behind a uuid-confirm + preview gate.

    Subclasses set `name`, `description`, `verb`, and `consequence`, and implement
    `_perform` (the SDK op) and `succeeded` (the success message). The gate itself —
    confirm-by-uuid, preview-on-refuse, error relay — lives here, once.

    A `PlatformTool`: the hosting agent binds the SDK client and current-timeline uuid
    before the loop runs. Until bound, `run` reports it is not connected (via
    `PlatformError`) rather than failing obscurely.
    """

    # --- subclass hooks ------------------------------------------------------
    #
    # `name` and `description` are the standard `Tool` contract (a loud, model-facing
    # description belongs on the subclass). The three below are this base's extra hooks.

    #: The verb used in every message — e.g. "lock" or "delete".
    verb: str = ""
    #: A one-line warning the refusal carries, naming the irreversible consequence.
    consequence: str = ""

    parameters = {
        "type": "object",
        "properties": {
            "confirm": {
                "type": "string",
                "description": (
                    "Required to actually proceed. It must EXACTLY equal the uuid of the "
                    "timeline you mean to affect — a deliberate, target-specific yes. Because "
                    "this is irreversible, a missing or mismatched confirm changes nothing: "
                    "you instead get a preview of exactly what would be affected and the uuid "
                    "to pass back."
                ),
            },
            "timeline": {
                "type": "string",
                "description": (
                    "Optional timeline uuid to act on instead of the current one. Omit to "
                    "target the timeline you are engaged on."
                ),
            },
        },
        "required": [],
    }

    def run(self, confirm: str | None = None, timeline: str | None = None) -> str:
        """Act on `timeline` (or the current one) — but only when `confirm` is its uuid."""
        target = timeline or self.context.timeline
        if confirm != target:
            return self._refuse(target, confirm)
        try:
            subject = self.context.client.timelines.get(target)
            self._perform(subject)
        except BaseCradleError as error:
            return f"Couldn't {self.verb} the timeline: {explain(error)}"
        return self.succeeded(subject)

    # --- the gate ------------------------------------------------------------

    def _refuse(self, target: str, confirm: str | None) -> str:
        """Preview-on-refuse: name what's at stake and hand back the exact confirm to use.

        Does one benign GET so the refusal can name the timeline and its item count. If even
        that read fails, refuse without a preview rather than letting the error escape —
        still naming the consequence and the uuid to confirm with.
        """
        try:
            subject = self.context.client.timelines.get(target)
        except BaseCradleError as error:
            return (
                f"Refused to {self.verb} timeline {target}: this is irreversible. "
                f"{self.consequence} To proceed, call {self.name} again with confirm={target}. "
                f"(Couldn't preview the timeline first: {explain(error)}.)"
            )
        if confirm is None:
            why = "No confirm was passed."
        else:
            why = f"The confirm you passed ({confirm!r}) does not match this timeline's uuid."
        return (
            f"Refused to {self.verb} timeline {subject.name!r} (uuid={subject.uuid}), which "
            f"currently has {len(subject.items)} item(s). {why} {self.consequence} If you "
            f"really mean to {self.verb} THIS timeline, call {self.name} again with "
            f"confirm={subject.uuid}."
        )

    # --- subclass operation --------------------------------------------------

    def _perform(self, timeline: Timeline) -> None:
        """Run the irreversible SDK op on the resolved timeline. Implemented by subclasses."""
        raise NotImplementedError

    def succeeded(self, timeline: Timeline) -> str:
        """The success message, naming what was done. Implemented by subclasses."""
        raise NotImplementedError
