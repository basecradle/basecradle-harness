"""The assets tool, against a respx-mocked platform.

No live platform call: respx stands in for the SDK's HTTP transport, returning
wire-shaped asset payloads (list page, single asset, create) and the blob bytes a
read downloads. The fictional cast: Nova Digital (``nova``, AI) is the agent;
files live on John Doe's timeline.

The tool is exercised through a *real* `BaseCradle` client — only its HTTP is
mocked — so these tests pin the tool against the SDK's true surface, the way the
constitution asks (mock at the transport level, never the live API).
"""

import httpx
import pytest
import respx
from basecradle import BaseCradle

from basecradle_harness import AssetsTool, PlatformContext, PlatformError

BC_URL = "https://basecradle.com"
FAKE_TOKEN = "bc_uat_KqI8zFxkQ0OZ8vYwT7mWcVtR3nSdLpEa"

NOVA_UUID = "019e7750-66ee-79c8-ad8a-bbb6ea7c2bcc"  # the agent
JOHN_UUID = "019e7750-66ee-7e50-9e54-3bf8c3d6a8f1"  # the human
TIMELINE_UUID = "019e7750-66ee-7f53-829f-13a8a710b6da"
OTHER_TIMELINE = "019e7760-1234-7abc-8def-0123456789ab"

# Well-formed UUIDv7 asset (content) uuids.
A_TEXT = "019e7751-4a1b-7c2d-8e3f-1a2b3c4d5e6f"
A_BIN = "019e7752-5b2c-7d3e-9f40-2b3c4d5e6f70"
A_BIG = "019e7753-6c3d-7e4f-8051-3c4d5e6f7081"

BLOB_URL = f"{BC_URL}/rails/active_storage/blobs/redirect/abc123/notes.md"


# --- wire payload builders ---------------------------------------------------


def asset(*, uuid, filename, content_type, byte_size, description="", url=BLOB_URL):
    """An asset in subject form (the SDK's documented shape)."""
    return {
        "type": "asset",
        "created_at": "2026-06-04T00:00:00.000Z",
        "user": {"uuid": JOHN_UUID, "handle": "john", "name": "John Doe", "kind": "human"},
        "timeline": {"uuid": TIMELINE_UUID},
        "content": {
            "uuid": uuid,
            "description": description,
            "file": {
                "filename": filename,
                "byte_size": byte_size,
                "content_type": content_type,
                "checksum": "Yp9p9C8m6Xv2qS1nKQ0r3w==",
                "url": url,
            },
        },
    }


def text_asset():
    return asset(
        uuid=A_TEXT,
        filename="notes.md",
        content_type="text/markdown",
        byte_size=21,
        description="Meeting notes",
    )


def binary_asset():
    return asset(
        uuid=A_BIN, filename="report.pdf", content_type="application/pdf", byte_size=184320
    )


def big_text_asset():
    return asset(uuid=A_BIG, filename="huge.log", content_type="text/plain", byte_size=2_000_000)


def _numbered_asset(i):
    """A distinct, well-formed asset payload for list-cap tests (varied node bits)."""
    node = (i * 0x9E3779B1) & 0xFFFFFFFFFFFF
    return asset(
        uuid=f"019e7758-1a2b-7c3d-8e4f-{node:012x}",
        filename=f"file{i}.txt",
        content_type="text/plain",
        byte_size=10,
    )


@pytest.fixture
def client():
    c = BaseCradle(token=FAKE_TOKEN)
    yield c
    c.close()


@pytest.fixture
def tool(client):
    """An AssetsTool bound to John's timeline through a real client."""
    t = AssetsTool()
    t.bind(PlatformContext(client=client, timeline=TIMELINE_UUID))
    return t


# --- list --------------------------------------------------------------------


def test_list_renders_each_asset_with_its_uuid(tool):
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/assets", params={"timeline": TIMELINE_UUID}).mock(
            return_value=httpx.Response(
                200, json={"assets": [text_asset(), binary_asset()], "next_cursor": None}
            )
        )
        result = tool.run(action="list")

    assert A_TEXT in result
    assert "notes.md" in result
    assert "Meeting notes" in result
    assert "report.pdf" in result


def test_list_on_an_empty_timeline_says_so(tool):
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/assets").mock(
            return_value=httpx.Response(200, json={"assets": [], "next_cursor": None})
        )
        assert "No files" in tool.run(action="list")


def test_list_renders_an_asset_with_no_description(tool):
    """An asset uploaded without a description omits the field — must not crash."""
    payload = text_asset()
    del payload["content"]["description"]  # the API omits it, not null
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/assets").mock(
            return_value=httpx.Response(200, json={"assets": [payload], "next_cursor": None})
        )
        result = tool.run(action="list")

    assert "notes.md" in result
    assert "Meeting notes" not in result  # nothing to render, and no trailing dash


def test_list_does_not_claim_more_at_exactly_the_limit(tool, monkeypatch):
    monkeypatch.setattr("basecradle_harness._assets.DEFAULT_LIST_LIMIT", 2)
    two = [_numbered_asset(0), _numbered_asset(1)]
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/assets").mock(
            return_value=httpx.Response(200, json={"assets": two, "next_cursor": None})
        )
        result = tool.run(action="list")

    assert "there may be more" not in result


def test_list_claims_more_past_the_limit(tool, monkeypatch):
    monkeypatch.setattr("basecradle_harness._assets.DEFAULT_LIST_LIMIT", 2)
    three = [_numbered_asset(i) for i in range(3)]
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/assets").mock(
            return_value=httpx.Response(200, json={"assets": three, "next_cursor": None})
        )
        result = tool.run(action="list")

    assert "there may be more" in result
    # The over-limit asset is not rendered; only the cap's worth is shown.
    assert _numbered_asset(2)["content"]["uuid"] not in result


# --- read --------------------------------------------------------------------


def test_read_inlines_decoded_text(tool):
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/assets/{A_TEXT}").mock(
            return_value=httpx.Response(200, json={"asset": text_asset()})
        )
        mock.get(BLOB_URL).mock(return_value=httpx.Response(200, content=b"# Notes\nship the tool"))
        result = tool.run(action="read", uuid=A_TEXT)

    assert "# Notes\nship the tool" in result
    assert "notes.md" in result  # the metadata header is present too


def test_read_describes_a_binary_file_without_dumping_bytes(tool):
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/assets/{A_BIN}").mock(
            return_value=httpx.Response(200, json={"asset": binary_asset()})
        )
        # No blob route: a binary read must not download.
        result = tool.run(action="read", uuid=A_BIN)

    assert "report.pdf" in result
    assert "not inlined" in result
    assert "binary" in result


def test_read_does_not_inline_oversized_text(tool):
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/assets/{A_BIG}").mock(
            return_value=httpx.Response(200, json={"asset": big_text_asset()})
        )
        # No blob route: oversized text is described, not fetched.
        result = tool.run(action="read", uuid=A_BIG)

    assert "huge.log" in result
    assert "not inlined" in result
    assert "inline limit" in result


def test_read_treats_undecodable_text_as_binary(tool):
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/assets/{A_TEXT}").mock(
            return_value=httpx.Response(200, json={"asset": text_asset()})
        )
        mock.get(BLOB_URL).mock(return_value=httpx.Response(200, content=b"\xff\xfe\x00bad"))
        result = tool.run(action="read", uuid=A_TEXT)

    assert "not valid UTF-8" in result


def test_read_without_a_uuid_is_a_friendly_error(tool):
    assert "needs the asset's uuid" in tool.run(action="read")


# --- create ------------------------------------------------------------------


def test_create_uploads_the_produced_text_as_a_named_file(tool):
    captured = {}

    def capture(request):
        captured["body"] = request.content
        return httpx.Response(
            201,
            json={
                "asset": asset(
                    uuid=A_TEXT, filename="summary.md", content_type="text/markdown", byte_size=12
                )
            },
        )

    with respx.mock(assert_all_called=True) as mock:
        # create resolves the timeline (public path), then POSTs the upload.
        mock.get(f"{BC_URL}/timelines/{TIMELINE_UUID}").mock(
            return_value=httpx.Response(200, json=_timeline_envelope())
        )
        mock.post(f"{BC_URL}/timelines/{TIMELINE_UUID}/assets").mock(side_effect=capture)
        result = tool.run(
            action="create",
            content="hello peers",
            filename="summary.md",
            description="A short summary",
        )

    assert "Uploaded 'summary.md'" in result
    # The multipart body carries the produced content and the filename.
    assert b"hello peers" in captured["body"]
    assert b"summary.md" in captured["body"]
    assert b"A short summary" in captured["body"]


def test_create_requires_content_and_filename(tool):
    assert "needs both" in tool.run(action="create", content="x")
    assert "needs both" in tool.run(action="create", filename="x.txt")


# --- cross-timeline + validation + binding -----------------------------------


def test_an_explicit_timeline_overrides_the_current_one(tool):
    with respx.mock(assert_all_called=True) as mock:
        route = mock.get(f"{BC_URL}/assets", params={"timeline": OTHER_TIMELINE}).mock(
            return_value=httpx.Response(200, json={"assets": [], "next_cursor": None})
        )
        tool.run(action="list", timeline=OTHER_TIMELINE)

    assert route.called  # it queried the other timeline, not the bound one


def test_unknown_action_is_reported(tool):
    assert "unknown action" in tool.run(action="delete")


def test_an_unbound_tool_raises_platform_error():
    with pytest.raises(PlatformError):
        AssetsTool().run(action="list")


# --- shared wire helper ------------------------------------------------------


def _timeline_envelope():
    return {
        "timeline": {
            "uuid": TIMELINE_UUID,
            "name": "Incident response",
            "locked": False,
            "created_at": "2026-06-01T00:00:00.000Z",
            "updated_at": "2026-06-02T00:00:00.000Z",
            "owner": {"uuid": JOHN_UUID, "handle": "john", "name": "John Doe", "kind": "human"},
            "participants": [
                {"uuid": NOVA_UUID, "handle": "nova", "name": "Nova Digital", "kind": "ai"}
            ],
        },
        "items": [],
    }
