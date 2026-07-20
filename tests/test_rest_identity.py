"""Every BaseCradle platform tool names its public REST identity (issue #334).

The point is orientation, not behavior: a model — including a weak one — reads a
tool's `description` at tool-choice time, so the one-line "this tool calls that same
endpoint" mapping has to ride *inside* the description, not only in a doc it may never
open. Cold-asked "what REST endpoint does your messages tool hit?", the agent should be
able to answer from the tool it is holding.

Two invariants, and the second is the founder-locked scope guard:

- **Every platform-resource tool carries the line, with its real route.** The route is
  byte-checked against the live ``docs/api.yaml`` (never invented), and the wording is
  *identity* — "this tool calls that same endpoint" — never analogy ("similar to").
- **Nothing else carries it.** MCP servers and vendor built-ins (search, code execution,
  media) and local tools (memory, web_fetch) are not BaseCradle REST resources, so they
  get no identity line — the same "platform tools only" boundary the handoff drew.

The guarded one-way actions (`lock`, `delete`) additionally state the line is **not a
bypass**: the confirm=uuid discipline holds on the REST endpoint exactly as it does in
the tool.
"""

from __future__ import annotations

import basecradle_harness as h

# Each platform tool → the REST route its description must name. The route is the
# create/primary one (a multi-action tool points at the docs anchor for the rest); the
# path-parameter naming (`{timeline_uuid}`) mirrors the human-facing docs/api.md and the
# handoff's copy pattern, and each route's existence is verified against docs/api.yaml.
PLATFORM_TOOL_ROUTES = {
    h.MessagesTool: "POST /timelines/{timeline_uuid}/messages",
    h.TasksTool: "POST /timelines/{timeline_uuid}/tasks",
    h.TimelinesTool: "POST /timelines",
    h.AssetsTool: "POST /timelines/{timeline_uuid}/assets",
    h.UsersTool: "GET /users",
    h.TrustTool: "POST /users/{user_uuid}/trust",
    h.WebhookEndpointsTool: "POST /timelines/{timeline_uuid}/webhook_endpoints",
    h.WebhookEventsTool: "GET /webhook_events",
    h.LockTool: "POST /timelines/{timeline_uuid}/lock",
    h.DeleteTool: "DELETE /timelines/{timeline_uuid}",
}

# The two guarded, irreversible timeline actions — their line must reaffirm the gate.
GUARDED_TOOLS = {h.LockTool, h.DeleteTool}


# Every other tool the package exports — vendor built-ins (media, search, code, shell),
# self-authorship, and local tools (memory, web_fetch, xai balance) — is NOT a BaseCradle
# REST resource and must carry no identity line. Derive the set from the exports rather
# than hand-listing it, so a tool added later can't silently outgrow the scope guard.
def _exported_tool_classes():
    """Every concrete `Tool` subclass the package exports (those with a real description)."""
    classes = []
    for name in dir(h):
        obj = getattr(h, name)
        if isinstance(obj, type) and issubclass(obj, h.Tool) and obj is not h.Tool:
            # Abstract bases (PlatformTool, ConfirmedTimelineAction) never assign a
            # description, so this keeps only the concrete, model-facing tools.
            if isinstance(getattr(obj, "description", None), str):
                classes.append(obj)
    return classes


NON_PLATFORM_TOOLS = [c for c in _exported_tool_classes() if c not in PLATFORM_TOOL_ROUTES]


def test_every_platform_tool_names_its_rest_route():
    for tool, route in PLATFORM_TOOL_ROUTES.items():
        description = tool.description
        # Identity, never analogy — the exact wording the handoff locked.
        expected = f"Platform REST: {route} — this tool calls that same endpoint"
        assert expected in description, f"{tool.__name__} missing/incorrect REST identity line"
        assert "similar to" not in description.lower(), f"{tool.__name__} used analogy wording"


def test_multi_action_tools_point_at_the_docs_anchor():
    # A multi-action tool names one route inline and sends the model to the tools↔HTTP
    # mapping anchor for the rest; the single-action guarded tools use the plain docs URL.
    for tool in PLATFORM_TOOL_ROUTES:
        if tool in GUARDED_TOOLS:
            continue
        assert "https://basecradle.com/docs/api.md#tools-and-the-http-api" in tool.description, (
            f"{tool.__name__} should point at the tools-and-the-http-api anchor"
        )


def test_guarded_tools_state_the_line_is_not_a_bypass():
    # lock/delete: naming the endpoint must not read as a way around confirm=uuid.
    for tool in GUARDED_TOOLS:
        description = tool.description
        assert "same confirm=uuid discipline (not a bypass)" in description, tool.__name__
        # The uuid-confirm gate the surrounding description establishes still stands.
        assert "confirm=<the timeline's uuid>" in description, tool.__name__


def test_non_platform_tools_carry_no_rest_identity_line():
    # The scope guard: MCP/vendor built-ins and local tools are not REST resources.
    # Guard against a vacuous pass — the derived set must actually contain the built-ins.
    names = {t.__name__ for t in NON_PLATFORM_TOOLS}
    assert {"WebFetchTool", "MemoryTool", "GenerateImageTool"} <= names, (
        "derived non-platform set looks wrong — refusing a vacuous scope-guard check"
    )
    for tool in NON_PLATFORM_TOOLS:
        assert "Platform REST:" not in tool.description, (
            f"{tool.__name__} is not a BaseCradle REST resource and must carry no identity line"
        )
