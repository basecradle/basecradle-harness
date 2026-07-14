"""The persistent operating brief: Turn 0, re-asserted on every wake.

Group 1 seeded a one-time onboarding orientation into a session's first turn — a
field-scrape of the structured Dashboard, composed once and then aging into the
distant past of a long transcript. This is its replacement: a **brief re-asserted
on every wake**, so the agent's standing operating context is always *recent* in
the conversation, not buried at turn 1.

The brief is composed, in order, of a current-time anchor followed by four parts:

0. **The current-time anchor** — `Current Time: <UTC> (+00:00, <weekday>)` plus a one-line
   UTC-conversion instruction, composed in `_wake.py::_now_line` and passed in fresh on every
   wake. It grounds the model in the absolute "now" (the brief is re-composed and re-injected
   each wake, so it is always current), is the reference every inbound item's `[created_at]`
   stamp is read against, and tells the model to convert UTC → a named locale before answering
   a local-time question (issue #180).
0b. **The step-budget statement** (`render_budget`) — the one-time "this turn has a budget of
   N steps, a live counter follows each step, per-turn and resets each wake" rule (issue #243),
   so the live per-step counter the engine injects can stay terse. Omitted when there is no
   budget to announce.
1. **`initialize.md`** — the framework's authored operating guidance: how to behave
   here, plus the cross-cutting gotchas the function schemas can't convey. Provider-
   independent (identical on every install).
2. **The generated tool manifest** — "Your active tools right now: …", rendered from
   Group 2's resolution (`ResolvedTools.manifest`). Always matches the active provider
   and the operator's drop-ins, so it can never drift from what the model can actually
   call. A tool's optional one-line `note` rides along.
3. **The live `dashboard.md`** — the platform's *maintained* primer (identity, surfaces,
   the concept map — including how trust works), fetched fresh from ``/users/dashboard.md``.
   A fetch failure degrades gracefully: the brief is composed without it, never broken.
4. **`system-prompt.md`** — the operator's personality charter.

Composition is pure (`compose_brief` / `render_manifest`); the one impure piece, the
live dashboard fetch (`fetch_dashboard_md`), is isolated and tolerant by construction.
"""

from __future__ import annotations

from collections.abc import Sequence


def render_manifest(entries: Sequence[tuple[str, str | None]]) -> str | None:
    """The "Your active tools right now" block, from ``(name, note)`` pairs.

    Each active tool is one line — its name, plus its optional one-line ``note`` after an
    em dash when present (a tool without one just lists its name). Returns ``None`` for an
    empty tool set, so the composer simply omits the section rather than emitting an empty
    heading.
    """
    if not entries:
        return None
    lines = ["Your active tools right now:"]
    for name, note in entries:
        lines.append(f"- {name} — {note}" if note else f"- {name}")
    return "\n".join(lines)


def render_safety(notices: Sequence[str] | None) -> str | None:
    """The safe-by-default opt-out block, from the resolved set's `notices`, or ``None``.

    Each notice is one line — an active MCP server, or a drop-in tool the locked policy
    refused (Group 5, Part B). Returns ``None`` for an empty/absent list, so a pure-Harness
    agent (no MCP, no policy-refused tool) composes exactly the brief it did before, with no
    safety section at all. When present, the block is headed so the agent reads it as the
    auditable "you have left the safe-by-default zone" marker the brief must not hide.
    """
    lines = [notice for notice in (notices or []) if notice and notice.strip()]
    if not lines:
        return None
    header = "⚠ Safe-by-default opt-out — this agent has loaded tools beyond the shipped safe set:"
    return "\n".join([header, *(f"- {line}" for line in lines)])


def render_defects(notices: Sequence[str] | None) -> str | None:
    """The broken-shipped-default defect block, from the resolved set's `broken`, or ``None``.

    Distinct from `render_safety`: a safe-by-default opt-out is something the operator *chose*,
    while a broken shipped default is a **defect** — a capability silently disabled by a stale
    overlay or a packaging bug (issue #160). The constitution forbids swallowing that quietly,
    so when present it is headed loudly and listed, one line per broken default, in the Turn-0
    brief — never mislabeled as an opt-out. Returns ``None`` for an empty/absent list, so a
    healthy agent composes exactly the brief it did before.
    """
    lines = [notice for notice in (notices or []) if notice and notice.strip()]
    if not lines:
        return None
    header = (
        "⚠ Tool Defect — a shipped default tool failed to load, so its capability is "
        "currently unavailable. If asked to use it, say so plainly. It is fixed by re-running "
        "basecradle-harness-install or repairing the file; if you cannot do that yourself, "
        "raise it on a timeline where someone who can will see it — nobody is watching your "
        "logs:"
    )
    return "\n".join([header, *(f"- {line}" for line in lines)])


def render_budget(max_steps: int | None) -> str | None:
    """The one-time step-budget statement for the persistent brief, or ``None``.

    States the rule once per wake — the turn has a budget of N steps, a live counter follows
    every step, and the budget is per-turn and resets each wake — so the per-step counter note
    can stay terse (issue #243). ``None``/non-positive → omitted (a caller with no budget to
    announce composes exactly the brief it did before). A sentence, so sentence case.

    It names the `messages` tool (issue #295). "Posted with a tool" leaves the model one inference
    short of the thing it must actually do, and the small-model cohort does not make it.
    """
    if max_steps is None or max_steps <= 0:
        return None
    return (
        f"Step budget: this turn runs for up to {max_steps} steps (one step = one of your "
        f"model turns, whether it calls tools or ends in text). A live counter — 'Step N of "
        f"{max_steps}' — is appended right before each step, so you always know where you "
        "stand. The budget is per-turn and resets on your next wake. Treat it as a hard "
        "constraint: end with plain text before it runs out — and remember that anything you "
        "mean a peer to see must be posted with the `messages` tool *before* then, since your "
        "closing text is unspoken. If work remains, schedule a follow-up task so the next turn "
        "continues it."
    )


def compose_brief(
    *,
    now: str | None = None,
    budget: str | None = None,
    initialize: str | None,
    manifest: str | None,
    defects: str | None = None,
    safety: str | None = None,
    dashboard: str | None,
    memory: str | None = None,
    system_prompt: str | None,
) -> str | None:
    """Join the brief parts in order, skipping any that are absent or empty.

    Order is load-bearing: the **current-time anchor** first (the absolute "now" every other
    item's age is reasoned against — `_wake.py::_now_line`), then operating guidance (how to
    act), then the tools the agent has, then any **tool defect** (a shipped default that failed
    to load — issue #160 — right after the manifest it contradicts, so the agent reads "you
    have these tools, but this one is broken" together), then the **safe-by-default opt-out
    notice** (Group 5), then the live dashboard (where it is), then any recalled **memory**
    relevant to the turn (the memory provider's `context` hook — injected just before the
    charter, the way middleware memory systems inject retrieved context before the system
    prompt), then the personality charter. Any part may be absent — a missing dashboard (fetch
    failed), a memory provider that recalled nothing, no MCP/policy opt-out, no broken default,
    an operator who blanked their charter — and the brief is composed from whatever remains.
    With nothing at all, returns ``None``.

    ``now``, ``budget``, ``defects``, ``safety``, and ``memory`` default to ``None`` so a caller
    with none of them (a test exercising composition, or the common no-MCP / default-SQLite-
    provider case) composes exactly the brief it did before these seams existed. The **step
    budget** rides right after the time anchor and before the operating guidance — it is a
    standing fact about how the turn is bounded, so the model reads it up front (issue #243).
    """
    parts = [
        part
        for part in (
            now,
            budget,
            initialize,
            manifest,
            defects,
            safety,
            dashboard,
            memory,
            system_prompt,
        )
        if part and part.strip()
    ]
    return "\n\n".join(parts) if parts else None


def fetch_dashboard_md(client: object) -> str | None:
    """The platform's live ``dashboard.md`` primer, or ``None`` on any failure (graceful).

    The structured ``GET /users/dashboard`` the SDK exposes as ``client.me`` is JSON; the
    *primer* the platform maintains for a freshly-woken peer is the Markdown at
    ``/users/dashboard.md``, which the SDK does not (yet) wrap with a typed accessor. So we
    fetch it over the SDK client's already-authenticated transport — reusing its base URL,
    token, and headers — rather than standing up a second HTTP stack with separate auth.

    **Never break the wake (the issue's hard requirement).** Every failure mode — the
    transport being absent, a non-2xx, a connection error, an empty body — degrades to
    ``None``, and the brief is composed without the dashboard section. A primer that briefly
    fails to load must never take an agent down with it.

    The ``.md`` path itself selects the Markdown representation, so no restrictive ``Accept``
    header is sent — a strict ``Accept: text/markdown`` would risk a ``406`` if the server
    negotiates differently, which would silently drop the dashboard section on every wake.
    """
    transport = getattr(client, "_client", None)
    if transport is None:
        return None
    try:
        response = transport.get("/users/dashboard.md")
        if not response.is_success:
            return None
        return response.text.strip() or None
    except Exception:  # noqa: BLE001 - a primer fetch must never break the wake; degrade to None
        return None
