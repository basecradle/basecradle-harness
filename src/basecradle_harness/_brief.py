"""The persistent operating brief: Turn 0, re-asserted on every wake.

Group 1 seeded a one-time onboarding orientation into a session's first turn — a
field-scrape of the structured Dashboard, composed once and then aging into the
distant past of a long transcript. This is its replacement: a **brief re-asserted
on every wake**, so the agent's standing operating context is always *recent* in
the conversation, not buried at turn 1.

The brief is composed, in order, of four parts:

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


def compose_brief(
    *,
    initialize: str | None,
    manifest: str | None,
    dashboard: str | None,
    memory: str | None = None,
    system_prompt: str | None,
) -> str | None:
    """Join the brief parts in order, skipping any that are absent or empty.

    Order is load-bearing: operating guidance first (how to act), then the tools the agent
    has, then the live dashboard (where it is), then any recalled **memory** relevant to the
    turn (the memory provider's `context` hook — injected just before the charter, the way
    middleware memory systems inject retrieved context before the system prompt), then the
    personality charter. Any part may be absent — a missing dashboard (fetch failed), a
    memory provider that recalled nothing, an operator who blanked their charter — and the
    brief is composed from whatever remains. With nothing at all, returns ``None``.

    ``memory`` defaults to ``None`` so a caller with no memory context (the common case, and
    the default SQLite provider whose `context` is a no-op) composes exactly the four-part
    brief it did before this seam existed.
    """
    parts = [
        part
        for part in (initialize, manifest, dashboard, memory, system_prompt)
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
