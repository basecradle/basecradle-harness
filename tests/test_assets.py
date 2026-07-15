"""The assets tool, against a respx-mocked platform.

No live platform call: respx stands in for the SDK's HTTP transport, returning
wire-shaped asset payloads (list page, single asset, create) and the blob bytes a
read downloads. The fictional cast: Nova Digital (``nova``, AI) is the agent;
files live on John Doe's timeline.

The tool is exercised through a *real* `BaseCradle` client — only its HTTP is
mocked — so these tests pin the tool against the SDK's true surface, the way the
constitution asks (mock at the transport level, never the live API).
"""

import base64
import logging

import httpx
import pytest
import respx
from basecradle import BaseCradle

from basecradle_harness import AssetsTool, PlatformContext, PlatformError, ToolCall, ToolResult
from basecradle_harness._assets import model_sees_images
from basecradle_harness._idempotency import create_kind
from basecradle_harness._mcp import McpImageStore
from basecradle_harness._unspoken import SpeechLedger

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
A_IMG = "019e7754-7d4e-7f50-8162-4d5e6f708192"

BLOB_URL = f"{BC_URL}/rails/active_storage/blobs/redirect/abc123/notes.md"
IMG_URL = f"{BC_URL}/rails/active_storage/blobs/redirect/img456/cat.png"
PNG_BYTES = b"\x89PNG\r\n\x1a\n fake pixels"


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


def image_asset(*, content_type="image/png", byte_size=None, filename="cat.png"):
    return asset(
        uuid=A_IMG,
        filename=filename,
        content_type=content_type,
        byte_size=byte_size if byte_size is not None else len(PNG_BYTES),
        url=IMG_URL,
    )


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


# --- tool description (issue #263) -------------------------------------------


def test_description_warns_assets_are_shared_and_never_editable():
    # A live agent used assets as a private file cabinet (18 uploads, duplicate-as-edit
    # workaround, one with live credentials visible to every viewer). The tool the model reads
    # must say plainly that assets are shared and permanent, and steer private/working files to
    # the agent's own storage — so the guidance rides with the tool, not only the brief.
    description = AssetsTool.description
    assert "shared with every viewer and can never be edited or deleted" in description
    assert "prefer your own storage for private or working files" in description


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


def test_read_of_an_image_points_at_the_view_action(tool):
    """A binary read of an image nudges the model toward 'view' so it can look."""
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/assets/{A_IMG}").mock(
            return_value=httpx.Response(200, json={"asset": image_asset()})
        )
        # No blob route: read still must not download a binary.
        result = tool.run(action="read", uuid=A_IMG)

    assert "view" in result
    assert "cat.png" in result


# --- view --------------------------------------------------------------------


def test_view_returns_the_image_as_a_data_url(tool):
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/assets/{A_IMG}").mock(
            return_value=httpx.Response(200, json={"asset": image_asset()})
        )
        mock.get(IMG_URL).mock(return_value=httpx.Response(200, content=PNG_BYTES))
        result = tool.run(action="view", uuid=A_IMG)

    assert isinstance(result, ToolResult)
    assert "cat.png" in result.text
    # The tool result is metadata only — it never narrates perception (issue #316). Whether the
    # pixels are actually *seen* is the engine's call (it gates on the model's vision and captions
    # the injected image turn), so the tool must not promise a view it cannot verify.
    assert "Looking at" not in result.text
    assert result.text.startswith("uuid=")  # the `_describe` metadata line, and nothing else
    assert "image/png" in result.text
    assert len(result.images) == 1
    image = result.images[0]
    assert image.alt == "cat.png"
    expected = "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode("ascii")
    assert image.url == expected


def test_view_rejects_a_non_image(tool):
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/assets/{A_BIN}").mock(
            return_value=httpx.Response(200, json={"asset": binary_asset()})
        )
        # No blob route: a non-image is refused without downloading.
        result = tool.run(action="view", uuid=A_BIN)

    assert isinstance(result, str)
    assert "not an image" in result


def test_view_rejects_an_unviewable_image_type(tool):
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/assets/{A_IMG}").mock(
            return_value=httpx.Response(
                200, json={"asset": image_asset(content_type="image/svg+xml", filename="x.svg")}
            )
        )
        result = tool.run(action="view", uuid=A_IMG)

    assert isinstance(result, str)
    assert "not viewable" in result


def test_view_rejects_an_oversized_image(tool):
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/assets/{A_IMG}").mock(
            return_value=httpx.Response(
                200, json={"asset": image_asset(byte_size=30 * 1024 * 1024)}
            )
        )
        # No blob route: an oversized image is refused without downloading.
        result = tool.run(action="view", uuid=A_IMG)

    assert isinstance(result, str)
    assert "too large" in result


def test_view_rejects_an_empty_image(tool):
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/assets/{A_IMG}").mock(
            return_value=httpx.Response(200, json={"asset": image_asset(byte_size=0)})
        )
        # No blob route: an empty file is refused without downloading.
        result = tool.run(action="view", uuid=A_IMG)

    assert isinstance(result, str)
    assert "empty file" in result


def test_view_without_a_uuid_is_a_friendly_error(tool):
    assert "needs the asset's uuid" in tool.run(action="view")


# --- view/read 'latest' (issue #161) -----------------------------------------


def test_view_latest_resolves_to_the_newest_asset(tool):
    """uuid='latest' views the most recent file — an image the agent just posted, no uuid handed it."""
    with respx.mock(assert_all_called=True) as mock:
        # The newest-first asset filter the resolver reads to find 'latest'.
        mock.get(f"{BC_URL}/assets", params={"timeline": TIMELINE_UUID}).mock(
            return_value=httpx.Response(
                200, json={"assets": [image_asset(), text_asset()], "next_cursor": None}
            )
        )
        # Then the normal view path fetches that asset by its resolved uuid and the blob.
        mock.get(f"{BC_URL}/assets/{A_IMG}").mock(
            return_value=httpx.Response(200, json={"asset": image_asset()})
        )
        mock.get(IMG_URL).mock(return_value=httpx.Response(200, content=PNG_BYTES))
        result = tool.run(action="view", uuid="latest")

    assert isinstance(result, ToolResult)
    assert "cat.png" in result.text
    assert len(result.images) == 1


def test_latest_is_case_insensitive(tool):
    """'LATEST' (any case, surrounding space) resolves the same as 'latest'."""
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/assets", params={"timeline": TIMELINE_UUID}).mock(
            return_value=httpx.Response(200, json={"assets": [image_asset()], "next_cursor": None})
        )
        mock.get(f"{BC_URL}/assets/{A_IMG}").mock(
            return_value=httpx.Response(200, json={"asset": image_asset()})
        )
        mock.get(IMG_URL).mock(return_value=httpx.Response(200, content=PNG_BYTES))
        result = tool.run(action="view", uuid="  LATEST ")

    assert isinstance(result, ToolResult)


def test_read_latest_resolves_to_the_newest_asset(tool):
    """uuid='latest' works for read too, not only view."""
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/assets", params={"timeline": TIMELINE_UUID}).mock(
            return_value=httpx.Response(200, json={"assets": [text_asset()], "next_cursor": None})
        )
        mock.get(f"{BC_URL}/assets/{A_TEXT}").mock(
            return_value=httpx.Response(200, json={"asset": text_asset()})
        )
        mock.get(BLOB_URL).mock(return_value=httpx.Response(200, content=b"# Notes\nlatest"))
        result = tool.run(action="read", uuid="latest")

    assert "# Notes\nlatest" in result


def test_view_latest_on_an_empty_timeline_says_so(tool):
    """'latest' against a timeline with no files is a clean message, not a crash or a bad fetch."""
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/assets", params={"timeline": TIMELINE_UUID}).mock(
            return_value=httpx.Response(200, json={"assets": [], "next_cursor": None})
        )
        # No /assets/{uuid} route: 'latest' resolves to nothing, so no fetch is attempted.
        result = tool.run(action="view", uuid="latest")

    assert isinstance(result, str)
    assert "No files" in result


def test_latest_honors_an_explicit_timeline(tool):
    """uuid='latest' resolves against the passed timeline, not only the bound current one."""
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/assets", params={"timeline": OTHER_TIMELINE}).mock(
            return_value=httpx.Response(200, json={"assets": [image_asset()], "next_cursor": None})
        )
        mock.get(f"{BC_URL}/assets/{A_IMG}").mock(
            return_value=httpx.Response(200, json={"asset": image_asset()})
        )
        mock.get(IMG_URL).mock(return_value=httpx.Response(200, content=PNG_BYTES))
        result = tool.run(action="view", uuid="latest", timeline=OTHER_TIMELINE)

    assert isinstance(result, ToolResult)


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


# --- post_image: "show me what you see" (issue #318) -------------------------


def _img_tool(client, store, speech=None):
    """An AssetsTool bound to John's timeline with a per-wake MCP image `store`."""
    t = AssetsTool()
    t.bind(PlatformContext(client=client, timeline=TIMELINE_UUID, mcp_images=store, speech=speech))
    return t


def _capture_upload_mock(mock, *, filename, content_type="image/png"):
    """Wire the timeline-resolve GET + the assets POST, capturing the multipart body."""
    captured = {}

    def capture(request):
        captured["body"] = request.content
        captured["headers"] = request.headers
        return httpx.Response(
            201,
            json={
                "asset": asset(
                    uuid=A_IMG,
                    filename=filename,
                    content_type=content_type,
                    byte_size=len(PNG_BYTES),
                    url=IMG_URL,
                )
            },
        )

    mock.get(f"{BC_URL}/timelines/{TIMELINE_UUID}").mock(
        return_value=httpx.Response(200, json=_timeline_envelope())
    )
    mock.post(f"{BC_URL}/timelines/{TIMELINE_UUID}/assets").mock(side_effect=capture)
    return captured


def test_post_image_uploads_a_stashed_capture_by_handle(client):
    store = McpImageStore()
    handle = store.stash("image/png", PNG_BYTES)
    speech = SpeechLedger()
    tool = _img_tool(client, store, speech=speech)

    with respx.mock(assert_all_called=True) as mock:
        captured = _capture_upload_mock(mock, filename="capture.png")
        result = tool.run(action="post_image", image=handle)

    assert "Posted 'capture.png'" in result
    assert PNG_BYTES in captured["body"]  # the stashed bytes were uploaded
    assert b"capture.png" in captured["body"]  # a default filename derived from the mime type
    # It records a visible act on the timeline (issue #293), like create/generate_image.
    assert speech.acts and speech.acts[-1][0] == "asset"


def test_post_image_defaults_to_the_latest_capture(client):
    store = McpImageStore()
    store.stash("image/png", b"old")
    store.stash("image/jpeg", PNG_BYTES)  # newest
    tool = _img_tool(client, store)

    with respx.mock(assert_all_called=True) as mock:
        captured = _capture_upload_mock(mock, filename="capture.jpg", content_type="image/jpeg")
        result = tool.run(action="post_image")  # no image ref → latest

    assert "Posted 'capture.jpg'" in result
    assert PNG_BYTES in captured["body"]  # the newest capture, not the older one
    assert b"capture.jpg" in captured["body"]  # jpeg → .jpg default name


def test_post_image_honors_an_explicit_filename_and_description(client):
    store = McpImageStore()
    store.stash("image/png", PNG_BYTES)
    tool = _img_tool(client, store)

    with respx.mock(assert_all_called=True) as mock:
        captured = _capture_upload_mock(mock, filename="homepage.png")
        tool.run(action="post_image", filename="homepage.png", description="the landing page")

    assert b"homepage.png" in captured["body"]
    assert b"the landing page" in captured["body"]


def test_post_image_carries_no_idempotency_key(client):
    # The bytes live only in the volatile store, so the create must never be re-issued by a
    # recovery — it is keyless, exactly like a generated image's upload (issue #318). Two proofs:
    # the header is absent on the upload, and the call is not a keyed create in the ordinal count.
    store = McpImageStore()
    store.stash("image/png", PNG_BYTES)
    tool = _img_tool(client, store)

    with respx.mock(assert_all_called=True) as mock:
        captured = _capture_upload_mock(mock, filename="capture.png")
        tool.run(action="post_image")

    assert "Idempotency-Key" not in captured["headers"]
    assert create_kind(ToolCall(id="c", name="assets", arguments={"action": "post_image"})) is None


def test_post_image_with_no_captures_is_a_clean_error(client):
    # An empty store, or no store bound at all (no MCP configured), both explain there is nothing
    # to post rather than crashing.
    empty = _img_tool(client, McpImageStore())
    assert "no captured images" in empty.run(action="post_image")

    unbound = AssetsTool()
    unbound.bind(PlatformContext(client=client, timeline=TIMELINE_UUID))  # mcp_images=None
    assert "no captured images" in unbound.run(action="post_image")


def test_post_image_unknown_reference_is_a_clean_error(client):
    store = McpImageStore()
    store.stash("image/png", PNG_BYTES)
    tool = _img_tool(client, store)
    result = tool.run(action="post_image", image="mcp-image-99")
    assert "no captured image 'mcp-image-99'" in result


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


# --- model_sees_images: the fail-open vision gate (issue #228) ----------------


class _Provider:
    """A provider stub whose ``supports_vision`` answer is whatever the test sets."""

    def __init__(self, answer):
        self._answer = answer

    def supports_vision(self):
        if isinstance(self._answer, Exception):
            raise self._answer
        return self._answer


def test_a_definite_no_vision_answer_withholds_the_image():
    assert model_sees_images(_Provider(False)) is False


def test_a_yes_vision_answer_shows_the_image():
    assert model_sees_images(_Provider(True)) is True


def test_an_unknown_vision_answer_fails_open():
    """``None`` is *unknown*, not "no vision" — the gate shows the image rather than withhold it."""
    assert model_sees_images(_Provider(None)) is True


def test_a_provider_without_the_capability_shows_the_image():
    """Most adapters answer nothing (OpenAI, xAI): an absent capability is not a refusal."""
    assert model_sees_images(object()) is True


def test_a_raising_capability_fails_open_and_is_logged(caplog):
    """A metadata read must never break a wake: a raising capability degrades to "show", loudly."""
    with caplog.at_level(logging.WARNING, logger="basecradle_harness"):
        assert model_sees_images(_Provider(RuntimeError("boom"))) is True
    assert any("vision capability" in r.getMessage() for r in caplog.records)


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
