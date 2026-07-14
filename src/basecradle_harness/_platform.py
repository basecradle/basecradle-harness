"""The platform-aware tool seam: how a tool gets a live BaseCradle handle.

`MemoryTool` is self-contained — it needs nothing from the platform. The first
tool that acts *on* BaseCradle (assets, and every later Phase-2 tranche: tasks,
participants, trust, lock, webhooks) needs two things a plain `Tool` never had:

1. the authenticated `basecradle.BaseCradle` SDK client (the body's voice), and
2. the uuid of the timeline the agent is currently engaged on (where to act by
   default).

Neither exists when the `Harness` is built — they belong to the hosting agent
(`TimelineAgent`/`WakeAgent`), which is constructed *after* the harness and its
tools. And the engine is deliberately platform-ignorant (it is the same loop for
the locked and unlocked profiles), so context cannot be threaded through
`Engine.run`. The seam resolves both: context is **bound onto the tool instance**,
out of band, once, before the loop runs.

The contract is small, to match the rest of the kit:

- `PlatformContext` — the live handle: client + current timeline (+ where temp
  files may live).
- `PlatformTool` — a `Tool` subclass that declares `requires = {BASECRADLE}` and
  receives a context via `bind`. Its `context` property is the one place a
  subclass reaches the client and timeline; calling it unbound raises a clear,
  model-readable error rather than an `AttributeError`.
- `bind_platform_tools` — what a hosting agent calls to wire every platform-aware
  tool in one pass.

Because a wake (or a poll loop) serves exactly one timeline per process, binding
once is correct; cross-timeline use is an explicit argument on the operation, not
a re-bind. This is the seam every Phase-2 tool reuses unchanged.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from basecradle_harness._exceptions import PlatformError
from basecradle_harness._idempotency import IdempotencyKeys
from basecradle_harness._policy import BASECRADLE
from basecradle_harness._tools import Tool
from basecradle_harness._unspoken import SpeechLedger

if TYPE_CHECKING:
    from basecradle import BaseCradle, BaseCradleError

    from basecradle_harness._code import CodeExecutionBridge


@dataclass(frozen=True)
class PlatformContext:
    """The live platform handle a `PlatformTool` acts through.

    Args:
        client: The authenticated `basecradle.BaseCradle` SDK client — the only
            way a Harness tool does platform I/O (never raw HTTP to the platform).
        timeline: The uuid of the timeline the agent is currently engaged on. A
            platform op defaults here; an explicit uuid overrides it for the rare
            cross-timeline case.
        home: The directory under which a tool may stage temp files (uploads and
            downloads), or `None`. This is the agent's `HARNESS_HOME`; confining
            scratch under it keeps the safe profile's I/O bounded and cleanable.
        code_bridge: The per-wake code-execution Asset bridge (`_code.py`), or
            `None` when code execution is not active. The `code_attach` tool reaches
            it here to stage a BaseCradle Asset into the executor; every other
            platform tool ignores it.
        speech: The wake's `SpeechLedger` (`_unspoken.py`), or `None` outside a wake.
            **A tool that acts visibly on a timeline records it here** — a message
            posted, an asset shared, a task made. It is not bookkeeping for its own
            sake: since the final-text auto-post was removed (issue #293), a tool call
            is the *only* way an agent speaks, so this ledger is the only truthful
            answer to "did the agent say anything this wake?" (the bookend's `posted=`)
            and "did it act at all this turn?" (what the mention informer turns on).
            Recording is a two-line courtesy in the tool that already knows; inferring
            it later would mean parsing tool arguments to tell a `create` from a `list`.
        keys: The wake's `IdempotencyKeys` minter (`_idempotency.py`), or `None` outside
            a wake (the poll loop, the library API, a test). **A create tool asks it for a
            key and passes that to the SDK** — and that key is what makes a killed wake's
            create safe to re-issue: the wake that recovers it mints the *same* key, so the
            platform returns the original record instead of creating a second one (issue
            #297). `None` → no key → the SDK sends no `Idempotency-Key` header and the
            create behaves exactly as it did before any of this existed, which is what keeps
            the seam free for every caller that has no recovery to do.
    """

    client: BaseCradle
    timeline: str
    home: Path | None = None
    code_bridge: CodeExecutionBridge | None = None
    speech: SpeechLedger | None = None
    keys: IdempotencyKeys | None = None


class PlatformTool(Tool):
    """A `Tool` that acts on BaseCradle through a bound `PlatformContext`.

    Subclasses implement `run` exactly as any tool does, and reach the SDK client
    and current timeline through `self.context`. They inherit
    `requires = {BASECRADLE}`, so a profile that forbade platform I/O would refuse
    them at registration — the shipped `locked()` profile permits it.

    The context is bound by the hosting agent after construction (see
    `bind_platform_tools`). Before that, `self.context` raises `PlatformError`,
    which the engine turns into a result the model can read and recover from —
    far better than an opaque `AttributeError`.
    """

    requires = frozenset({BASECRADLE})

    _context: PlatformContext | None = None

    def bind(self, context: PlatformContext) -> None:
        """Attach the live platform handle. Called once by the hosting agent."""
        self._context = context

    @property
    def bound(self) -> bool:
        """Whether a `PlatformContext` has been bound yet."""
        return self._context is not None

    @property
    def context(self) -> PlatformContext:
        """The bound context, or `PlatformError` if the tool was never wired.

        Subclasses read `self.context.client` and `self.context.timeline` here.
        """
        if self._context is None:
            raise PlatformError(
                f"Tool {self.name!r} needs the platform but is not connected to one. "
                "It must run inside a TimelineAgent/WakeAgent (or have a PlatformContext "
                "bound) before it can act on BaseCradle."
            )
        return self._context

    # --- idempotency: a create a killed wake can safely re-issue (issue #297) ---

    def key(self, kind: str) -> str | None:
        """The `Idempotency-Key` for the platform create this tool is about to make.

        ``None`` when no minter is bound (the poll loop, the library API) — which the SDK reads as
        "send no header", so the create is exactly what it always was.

        **Ask for it once, at the top of the create branch, before validating anything.** The
        ordinal it encodes is counted off the transcript, and the transcript records the *call*, so
        a call that mints no key where the recovery expects one would shift every later ordinal in
        the turn by one — and an ordinal that is off by one is a key the platform has never seen,
        which is a message posted twice.
        """
        keys = self.context.keys
        return keys.mint(kind) if keys is not None else None

    # --- the speech ledger: say what you did, so the wake need not guess (issue #293) ---

    def spoke(self, message: object) -> None:
        """Record a message this tool just posted. The agent's only way of speaking."""
        self._record("spoke", message)

    def acted(self, kind: str, uuid: str | None = None) -> None:
        """Record a non-message action that lands visibly on a timeline (an asset, a task)."""
        self._record("acted", kind, uuid)

    def _record(self, method: str, *args: object) -> None:
        """Write to the bound `SpeechLedger` if there is one — a no-op when there isn't.

        Silently absent outside a wake (a bare `PlatformContext` in a test, a library embedder
        wiring tools by hand), because the ledger answers a *wake's* questions. A tool must never
        fail an action it already completed over a bookkeeping seam that isn't there, so this is
        the one place the `None` is handled, rather than at four call sites.
        """
        speech = getattr(self._context, "speech", None)
        if speech is not None:
            getattr(speech, method)(*args)


def explain(error: BaseCradleError) -> str:
    """The most human-readable string a platform (SDK) error carries.

    API errors are RFC 9457 problem documents: `detail` is the human sentence, with `title`
    and the raw message as fallbacks. This is what turns a refused platform action into an
    explanation the agent can relay — a tool's "Couldn't …" message, or a wake's degrade
    note — rather than a raw traceback. Shared by every place that surfaces a `basecradle`
    error to the model, so the fallback precedence lives in exactly one spot.
    """
    return error.detail or error.title or str(error)


def bind_platform_tools(tools: Iterable[Tool], context: PlatformContext) -> int:
    """Bind `context` into every `PlatformTool` in `tools`. Returns how many were bound.

    A hosting agent calls this once it knows its client and current timeline, so
    every platform-aware tool the harness holds is wired in a single pass. Plain
    tools (e.g. `MemoryTool`) are skipped untouched.
    """
    bound = 0
    for tool in tools:
        if isinstance(tool, PlatformTool):
            tool.bind(context)
            bound += 1
    return bound
