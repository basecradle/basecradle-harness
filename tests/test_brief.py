"""The persistent operating brief — composition, manifest rendering, and the live fetch.

Pure composition (`compose_brief`, `render_manifest`) is asserted directly; the one impure
piece, `fetch_dashboard_md`, is driven against a respx-mocked BaseCradle transport so the
graceful-degradation contract (never break the wake) is pinned without touching the network.
"""

from __future__ import annotations

import httpx
import respx
from basecradle import BaseCradle

from basecradle_harness import compose_brief, fetch_dashboard_md, render_manifest

BC_URL = "https://basecradle.com"
FAKE_TOKEN = "bc_uat_KqI8zFxkQ0OZ8vYwT7mWcVtR3nSdLpEa"


# --- render_manifest ----------------------------------------------------------


def test_render_manifest_lists_names_and_notes():
    text = render_manifest(
        [("memory", None), ("lock", "irreversible; confirm must equal the uuid.")]
    )
    assert text.splitlines() == [
        "Your active tools right now:",
        "- memory",
        "- lock — irreversible; confirm must equal the uuid.",
    ]


def test_render_manifest_is_none_when_empty():
    # No active tools → no heading, so the composer omits the section entirely.
    assert render_manifest([]) is None


# --- compose_brief ------------------------------------------------------------


def test_compose_brief_orders_the_four_parts():
    brief = compose_brief(
        initialize="INIT",
        manifest="MANIFEST",
        dashboard="DASH",
        system_prompt="CHARTER",
    )
    assert brief == "INIT\n\nMANIFEST\n\nDASH\n\nCHARTER"


def test_compose_brief_places_the_now_anchor_first():
    # The current-time anchor leads the brief, ahead of every other part, so the model
    # reads "now" before anything whose age it must reason about.
    brief = compose_brief(
        now="Current Time: 2026-06-21 17:09:49 UTC (Sunday)",
        initialize="INIT",
        manifest="MANIFEST",
        dashboard="DASH",
        system_prompt="CHARTER",
    )
    assert brief == (
        "Current Time: 2026-06-21 17:09:49 UTC (Sunday)\n\nINIT\n\nMANIFEST\n\nDASH\n\nCHARTER"
    )


def test_compose_brief_omits_the_now_anchor_when_absent():
    # `now` defaults to None, so a caller that passes none composes exactly as before.
    brief = compose_brief(
        now=None,
        initialize="INIT",
        manifest="MANIFEST",
        dashboard="DASH",
        system_prompt="CHARTER",
    )
    assert brief == "INIT\n\nMANIFEST\n\nDASH\n\nCHARTER"


def test_compose_brief_skips_absent_and_blank_parts():
    # A failed dashboard fetch (None) and a blanked charter (whitespace) both drop out, and
    # the brief is composed from what remains, in order — never a dangling blank section.
    brief = compose_brief(
        initialize="INIT", manifest="MANIFEST", dashboard=None, system_prompt="   "
    )
    assert brief == "INIT\n\nMANIFEST"


def test_compose_brief_is_none_when_nothing_to_say():
    assert compose_brief(initialize=None, manifest=None, dashboard=None, system_prompt=None) is None


# --- fetch_dashboard_md -------------------------------------------------------


def test_fetch_dashboard_md_returns_the_live_primer():
    with respx.mock(base_url=BC_URL) as router:
        router.get("/users/dashboard.md").mock(
            return_value=httpx.Response(200, text="# Welcome\n\nTrust is mutual at the gate.\n")
        )
        client = BaseCradle(token=FAKE_TOKEN)
        text = fetch_dashboard_md(client)
    assert text == "# Welcome\n\nTrust is mutual at the gate."  # fetched and trimmed


def test_fetch_dashboard_md_rides_the_authenticated_transport():
    with respx.mock(base_url=BC_URL) as router:
        route = router.get("/users/dashboard.md").mock(
            return_value=httpx.Response(200, text="primer")
        )
        client = BaseCradle(token=FAKE_TOKEN)
        fetch_dashboard_md(client)
    # The fetch reused the SDK client's own auth, not a second HTTP stack.
    assert route.calls.last.request.headers["Authorization"] == f"Bearer {FAKE_TOKEN}"


def test_fetch_dashboard_md_degrades_on_a_non_2xx():
    with respx.mock(base_url=BC_URL) as router:
        router.get("/users/dashboard.md").mock(return_value=httpx.Response(503))
        client = BaseCradle(token=FAKE_TOKEN)
        assert fetch_dashboard_md(client) is None  # never raises — the wake survives


def test_fetch_dashboard_md_degrades_on_a_connection_error():
    with respx.mock(base_url=BC_URL) as router:
        router.get("/users/dashboard.md").mock(side_effect=httpx.ConnectError("down"))
        client = BaseCradle(token=FAKE_TOKEN)
        assert fetch_dashboard_md(client) is None


def test_fetch_dashboard_md_degrades_on_an_empty_body():
    with respx.mock(base_url=BC_URL) as router:
        router.get("/users/dashboard.md").mock(return_value=httpx.Response(200, text="   "))
        client = BaseCradle(token=FAKE_TOKEN)
        assert fetch_dashboard_md(client) is None  # blank primer → omit, not an empty section


def test_fetch_dashboard_md_tolerates_a_transportless_client():
    # An object that is not an SDK client (no `_client`) degrades to None rather than raising.
    assert fetch_dashboard_md(object()) is None
