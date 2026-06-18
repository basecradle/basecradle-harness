"""The grok media tools (image + video), against a respx-mocked xAI API and platform.

No live calls: respx stands in for xAI's Images/Videos endpoints and for the BaseCradle SDK
transport (which takes the upload). The fictional cast: Nova Digital (``nova``, AI) generates
an image/video onto John Doe's timeline.

These tests assert the harness's half of the contract — the params sent, the async poll loop,
the legible error relay, and that the uploaded Asset's filename extension follows the *real*
bytes (sniffed). The ground-truth checks the handoff calls for (the posted Asset's actual
pixels / content-type, a real measured-dimension video, Live Search citations) are the
capital's live @jt verification on Eddie, which cannot run offline against a mock.
"""

import base64
import json

import httpx
import pytest
import respx
from basecradle import BaseCradle

from basecradle_harness import (
    GrokGenerateImageTool,
    GrokGenerateVideoTool,
    PlatformContext,
)

BC_URL = "https://basecradle.com"
XAI_BASE = "https://api.x.ai.test/v1"
IMAGES_URL = f"{XAI_BASE}/images/generations"
VIDEOS_URL = f"{XAI_BASE}/videos/generations"
FAKE_TOKEN = "bc_uat_KqI8zFxkQ0OZ8vYwT7mWcVtR3nSdLpEa"
FAKE_KEY = "xai-test-0123456789abcdefghijklmnop"

NOVA_UUID = "019e7750-66ee-79c8-ad8a-bbb6ea7c2bcc"
JOHN_UUID = "019e7750-66ee-7e50-9e54-3bf8c3d6a8f1"
TIMELINE_UUID = "019e7750-66ee-7f53-829f-13a8a710b6da"
A_MEDIA = "019e7754-7d4e-7f50-8162-4d5e6f708192"
SOURCE_UUID = "019e7754-7d4e-7f50-8162-aaaabbbbcccc"
REQUEST_ID = "019e7755-1c2d-7a3e-9b4f-5c6d7e8f9012"

JPEG_BYTES = b"\xff\xd8\xff\xe0 grok pixels"
MP4_BYTES = b"\x00\x00\x00\x18ftypmp42 grok video frames"


# --- shared fixtures / response builders -------------------------------------


@pytest.fixture
def client():
    c = BaseCradle(token=FAKE_TOKEN)
    yield c
    c.close()


@pytest.fixture
def image_tool(client):
    t = GrokGenerateImageTool(api_key=FAKE_KEY, base_url=XAI_BASE)
    t.bind(PlatformContext(client=client, timeline=TIMELINE_UUID))
    return t


@pytest.fixture
def video_tool(client):
    # poll_interval=0 so the async poll loop runs instantly under test.
    t = GrokGenerateVideoTool(api_key=FAKE_KEY, base_url=XAI_BASE, poll_interval=0)
    t.bind(PlatformContext(client=client, timeline=TIMELINE_UUID))
    return t


def image_b64(data=JPEG_BYTES):
    return {"data": [{"b64_json": base64.b64encode(data).decode("ascii")}]}


def image_url_body():
    return {"data": [{"url": f"{XAI_BASE}/blobs/generated.jpg"}]}


def asset_response(*, filename, content_type):
    return {
        "asset": {
            "type": "asset",
            "created_at": "2026-06-17T00:00:00.000Z",
            "user": {"uuid": NOVA_UUID, "handle": "nova", "name": "Nova Digital", "kind": "ai"},
            "timeline": {"uuid": TIMELINE_UUID},
            "content": {
                "uuid": A_MEDIA,
                "description": "Generated media",
                "file": {
                    "filename": filename,
                    "byte_size": 99,
                    "content_type": content_type,
                    "checksum": "Yp9p9C8m6Xv2qS1nKQ0r3w==",
                    "url": f"{BC_URL}/blobs/{A_MEDIA}",
                },
            },
        }
    }


def _timeline_envelope():
    return {
        "timeline": {
            "uuid": TIMELINE_UUID,
            "name": "Studio",
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


def source_asset_response(uuid):
    return {
        "asset": {
            "type": "asset",
            "created_at": "2026-06-04T00:00:00.000Z",
            "user": {"uuid": JOHN_UUID, "handle": "john", "name": "John Doe", "kind": "human"},
            "timeline": {"uuid": TIMELINE_UUID},
            "content": {
                "uuid": uuid,
                "description": "A source image",
                "file": {
                    "filename": "source.png",
                    "byte_size": 32,
                    "content_type": "image/png",
                    "checksum": "Yp9p9C8m6Xv2qS1nKQ0r3w==",
                    "url": f"{BC_URL}/blobs/{uuid}",
                },
            },
        }
    }


def _mock_upload(mock, captured):
    def upload(request):
        captured["upload"] = request.content
        # Echo the posted filename back so the test can assert the extension that was sent.
        name = _multipart_filename(request.content)
        captured["filename"] = name
        return httpx.Response(201, json=asset_response(filename=name, content_type="image/jpeg"))

    mock.get(f"{BC_URL}/timelines/{TIMELINE_UUID}").mock(
        return_value=httpx.Response(200, json=_timeline_envelope())
    )
    mock.post(f"{BC_URL}/timelines/{TIMELINE_UUID}/assets").mock(side_effect=upload)


def _multipart_filename(body: bytes) -> str:
    marker = b'filename="'
    start = body.find(marker)
    if start == -1:
        return ""
    start += len(marker)
    return body[start : body.find(b'"', start)].decode("ascii", "replace")


# --- grok_generate_image -----------------------------------------------------


def test_image_generates_and_posts_with_jpeg_extension_from_sniffed_bytes(image_tool):
    captured = {}
    with respx.mock(assert_all_called=True) as mock:
        gen = mock.post(IMAGES_URL).mock(return_value=httpx.Response(200, json=image_b64()))
        _mock_upload(mock, captured)
        result = image_tool.run(prompt="a neon Dallas skyline at dusk")

    sent = json.loads(gen.calls.last.request.content)
    assert sent["model"] == "grok-imagine-image-quality"
    assert sent["prompt"] == "a neon Dallas skyline at dusk"
    assert sent["response_format"] == "b64_json"  # we ask for bytes inline
    assert "aspect_ratio" not in sent and "resolution" not in sent  # omitted when unset
    assert JPEG_BYTES in captured["upload"]
    # The extension follows the *real* bytes (JPEG magic) regardless of any format knob.
    assert captured["filename"].endswith(".jpg")
    assert "Generated and posted" in result and A_MEDIA in result


def test_image_passes_aspect_ratio_and_resolution_only_when_set(image_tool):
    captured = {}
    with respx.mock(assert_all_called=True) as mock:
        gen = mock.post(IMAGES_URL).mock(return_value=httpx.Response(200, json=image_b64()))
        _mock_upload(mock, captured)
        image_tool.run(prompt="a cat", aspect_ratio="16:9", resolution="720p")

    sent = json.loads(gen.calls.last.request.content)
    assert sent["aspect_ratio"] == "16:9"
    assert sent["resolution"] == "720p"


def test_image_falls_back_to_url_response_and_downloads_it(image_tool):
    captured = {}
    with respx.mock(assert_all_called=True) as mock:
        mock.post(IMAGES_URL).mock(return_value=httpx.Response(200, json=image_url_body()))
        # The url-encoded result is fetched from the blob URL the API returned.
        mock.get(f"{XAI_BASE}/blobs/generated.jpg").mock(
            return_value=httpx.Response(200, content=JPEG_BYTES)
        )
        _mock_upload(mock, captured)
        result = image_tool.run(prompt="a fox")

    assert JPEG_BYTES in captured["upload"]
    assert A_MEDIA in result


def test_image_relays_the_real_xai_error_not_a_generic_400(image_tool):
    body = {"error": {"message": "aspect_ratio '5:5' is not supported"}}
    with respx.mock(assert_all_called=True) as mock:
        mock.post(IMAGES_URL).mock(return_value=httpx.Response(400, json=body))
        result = image_tool.run(prompt="x", aspect_ratio="5:5")

    assert "the xAI image API rejected the request" in result
    assert "aspect_ratio '5:5' is not supported" in result  # the true cause, relayed


def test_image_needs_a_prompt(image_tool):
    assert "needs a 'prompt'" in image_tool.run(prompt="   ")


# --- grok_generate_video (async submit + poll) -------------------------------


def test_video_submits_polls_to_done_and_posts_the_clip(video_tool):
    captured = {}
    with respx.mock(assert_all_called=True) as mock:
        submit = mock.post(VIDEOS_URL).mock(
            return_value=httpx.Response(200, json={"request_id": REQUEST_ID})
        )
        # First poll pending, second done — the loop must keep going until it resolves.
        mock.get(f"{XAI_BASE}/videos/{REQUEST_ID}").mock(
            side_effect=[
                httpx.Response(200, json={"status": "pending"}),
                httpx.Response(
                    200,
                    json={
                        "status": "done",
                        "video": {"url": f"{XAI_BASE}/clips/out.mp4", "duration": 6},
                    },
                ),
            ]
        )
        mock.get(f"{XAI_BASE}/clips/out.mp4").mock(
            return_value=httpx.Response(200, content=MP4_BYTES)
        )
        _mock_upload(mock, captured)
        result = video_tool.run(prompt="a drone shot over the ocean", duration=6)

    sent = json.loads(submit.calls.last.request.content)
    assert sent["model"] == "grok-imagine-video"
    assert sent["prompt"] == "a drone shot over the ocean"
    assert sent["duration"] == 6
    assert MP4_BYTES in captured["upload"]
    assert captured["filename"].endswith(".mp4")  # sniffed ftyp box → .mp4
    assert "Generated and posted" in result and A_MEDIA in result


def test_video_image_to_video_resolves_the_source_asset_to_a_url(video_tool):
    captured = {}
    with respx.mock(assert_all_called=True) as mock:
        # The source Asset uuid is resolved to its blob URL and sent as image_url.
        mock.get(f"{BC_URL}/assets/{SOURCE_UUID}").mock(
            return_value=httpx.Response(200, json=source_asset_response(SOURCE_UUID))
        )
        submit = mock.post(VIDEOS_URL).mock(
            return_value=httpx.Response(200, json={"request_id": REQUEST_ID})
        )
        mock.get(f"{XAI_BASE}/videos/{REQUEST_ID}").mock(
            return_value=httpx.Response(
                200, json={"status": "done", "video": {"url": f"{XAI_BASE}/clips/out.mp4"}}
            )
        )
        mock.get(f"{XAI_BASE}/clips/out.mp4").mock(
            return_value=httpx.Response(200, content=MP4_BYTES)
        )
        _mock_upload(mock, captured)
        video_tool.run(prompt="animate this", image=SOURCE_UUID)

    sent = json.loads(submit.calls.last.request.content)
    assert sent["image_url"] == f"{BC_URL}/blobs/{SOURCE_UUID}"


def test_video_relays_a_failed_job_status_legibly(video_tool):
    with respx.mock(assert_all_called=True) as mock:
        mock.post(VIDEOS_URL).mock(
            return_value=httpx.Response(200, json={"request_id": REQUEST_ID})
        )
        mock.get(f"{XAI_BASE}/videos/{REQUEST_ID}").mock(
            return_value=httpx.Response(
                200,
                json={"status": "failed", "error": {"message": "prompt violates policy"}},
            )
        )
        result = video_tool.run(prompt="something disallowed")

    assert "the xAI video API failed" in result
    assert "prompt violates policy" in result


def test_video_relays_a_submit_error(video_tool):
    with respx.mock(assert_all_called=True) as mock:
        mock.post(VIDEOS_URL).mock(
            return_value=httpx.Response(400, json={"error": {"message": "duration must be 1-15"}})
        )
        result = video_tool.run(prompt="x", duration=99)

    assert "the xAI video API rejected the request" in result
    assert "duration must be 1-15" in result


def test_video_needs_a_prompt(video_tool):
    assert "needs a 'prompt'" in video_tool.run(prompt="")
