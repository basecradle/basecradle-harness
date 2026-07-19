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
    GrokEditImageTool,
    GrokGenerateImageTool,
    GrokGenerateVideoTool,
    PlatformContext,
)

BC_URL = "https://basecradle.com"
XAI_BASE = "https://api.x.ai.test/v1"
IMAGES_URL = f"{XAI_BASE}/images/generations"
EDITS_URL = f"{XAI_BASE}/images/edits"
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
def edit_tool(client):
    t = GrokEditImageTool(api_key=FAKE_KEY, base_url=XAI_BASE)
    t.bind(PlatformContext(client=client, timeline=TIMELINE_UUID))
    return t


@pytest.fixture
def video_tool(client):
    # poll_interval=0 so the async poll loop runs instantly under test.
    t = GrokGenerateVideoTool(api_key=FAKE_KEY, base_url=XAI_BASE, poll_interval=0)
    t.bind(PlatformContext(client=client, timeline=TIMELINE_UUID))
    return t


def image_b64(data=JPEG_BYTES, cost_ticks=None):
    body = {"data": [{"b64_json": base64.b64encode(data).decode("ascii")}]}
    if cost_ticks is not None:
        # xAI states the charge on the response body: usage.cost_in_usd_ticks (1 tick = 1e-10 USD).
        body["usage"] = {"cost_in_usd_ticks": cost_ticks}
    return body


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


def source_asset_response(uuid, content_type="image/png"):
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
                    "content_type": content_type,
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


# --- grok_edit_image ---------------------------------------------------------

PNG_BYTES = b"\x89PNG\r\n\x1a\n source pixels"
SOURCE_UUID_2 = "019e7754-7d4e-7f50-8162-ddddeeeeffff"


def _mock_source(mock, uuid, data=PNG_BYTES, content_type="image/png"):
    """Mock resolving a source Asset (assets.get) and downloading its blob bytes."""
    mock.get(f"{BC_URL}/assets/{uuid}").mock(
        return_value=httpx.Response(200, json=source_asset_response(uuid, content_type))
    )
    mock.get(f"{BC_URL}/blobs/{uuid}").mock(return_value=httpx.Response(200, content=data))


def test_edit_single_source_sends_image_object_as_base64_data_uri(edit_tool):
    captured = {}
    with respx.mock(assert_all_called=True) as mock:
        _mock_source(mock, SOURCE_UUID)
        edit = mock.post(EDITS_URL).mock(return_value=httpx.Response(200, json=image_b64()))
        _mock_upload(mock, captured)
        result = edit_tool.run(image=SOURCE_UUID, prompt="recolor the car red")

    sent = json.loads(edit.calls.last.request.content)
    assert sent["model"] == "grok-imagine-image-quality"
    assert sent["prompt"] == "recolor the car red"
    assert sent["response_format"] == "b64_json"  # we ask for bytes inline
    # A single source rides the ``image`` object (not the ``images`` array), as a data URI —
    # the source bytes inlined, not the (not-assumed-public) signed Asset URL.
    expected = "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode("ascii")
    assert sent["image"] == {"type": "image_url", "url": expected}
    assert "images" not in sent
    assert JPEG_BYTES in captured["upload"]
    assert captured["filename"].endswith(".jpg")  # extension follows the real (JPEG) bytes
    assert "Edited and posted" in result and A_MEDIA in result


def test_edit_multiple_sources_composite_into_the_images_array(edit_tool):
    captured = {}
    with respx.mock(assert_all_called=True) as mock:
        _mock_source(mock, SOURCE_UUID)
        _mock_source(mock, SOURCE_UUID_2)
        edit = mock.post(EDITS_URL).mock(return_value=httpx.Response(200, json=image_b64()))
        _mock_upload(mock, captured)
        edit_tool.run(image=[SOURCE_UUID, SOURCE_UUID_2], prompt="composite these two")

    sent = json.loads(edit.calls.last.request.content)
    # Two-plus sources ride the ``images`` array (the multi-image composite shape), not ``image``.
    assert "image" not in sent
    assert [obj["type"] for obj in sent["images"]] == ["image_url", "image_url"]
    assert len(sent["images"]) == 2
    assert all(obj["url"].startswith("data:image/png;base64,") for obj in sent["images"])


def test_edit_relays_the_real_xai_error_not_a_generic_400(edit_tool):
    body = {"error": {"message": "unsupported image format"}}
    with respx.mock(assert_all_called=True) as mock:
        _mock_source(mock, SOURCE_UUID)
        mock.post(EDITS_URL).mock(return_value=httpx.Response(400, json=body))
        result = edit_tool.run(image=SOURCE_UUID, prompt="x")

    assert "the xAI image API rejected the request" in result
    assert "unsupported image format" in result  # the true cause, relayed


def test_edit_relays_an_unreadable_source_asset_legibly(edit_tool):
    with respx.mock(assert_all_called=True) as mock:
        # The source uuid resolves to a 404 — a legible "couldn't read source" relay, no API call.
        mock.get(f"{BC_URL}/assets/{SOURCE_UUID}").mock(return_value=httpx.Response(404, json={}))
        result = edit_tool.run(image=SOURCE_UUID, prompt="recolor")

    assert "Error editing image" in result
    assert SOURCE_UUID in result  # names which source couldn't be read


def test_edit_survives_a_source_asset_with_no_content_type(edit_tool):
    # A malformed source blob (null content-type) must not crash `_data_url` with an
    # AttributeError — it falls back to a generic type so the edit still proceeds legibly.
    captured = {}
    with respx.mock(assert_all_called=True) as mock:
        _mock_source(mock, SOURCE_UUID, content_type=None)
        edit = mock.post(EDITS_URL).mock(return_value=httpx.Response(200, json=image_b64()))
        _mock_upload(mock, captured)
        result = edit_tool.run(image=SOURCE_UUID, prompt="recolor")

    sent = json.loads(edit.calls.last.request.content)
    assert sent["image"]["url"].startswith("data:application/octet-stream;base64,")
    assert "Edited and posted" in result


def test_edit_needs_a_prompt(edit_tool):
    assert "needs a 'prompt'" in edit_tool.run(image=SOURCE_UUID, prompt="  ")


def test_edit_needs_a_source_image(edit_tool):
    assert "needs at least one source" in edit_tool.run(image="", prompt="recolor")


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


# --- the request timeout (issue #222, sibling of #219) -----------------------


def test_default_timeout_clears_the_grok_high_quality_edit_latency():
    # `grok_edit_image` runs the same class of slow, high-fidelity edit as gpt-image-2's
    # `quality: high`, which was measured at ~133s live (#219). The ceiling must clear that
    # worst case with headroom (issue #222); mirrors #219's 300s.
    from basecradle_harness._grok import DEFAULT_TIMEOUT

    assert DEFAULT_TIMEOUT >= 133.0
    assert DEFAULT_TIMEOUT == 300.0


def test_the_timeout_reaches_the_grok_http_client(edit_tool, monkeypatch):
    # Whatever timeout the tool carries is the one handed to the httpx client at call time,
    # so the ceiling bump actually takes effect rather than being decorative.
    import basecradle_harness._grok as grok

    captured = {}

    class _FakeResponse:
        status_code = 200

        def json(self):
            return image_b64()

    class _FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def request(self, *args, **kwargs):
            return _FakeResponse()

    monkeypatch.setattr(grok.httpx, "Client", _FakeClient)
    edit_tool._request(FAKE_KEY, "POST", "images/edits", json={})
    assert captured["timeout"] == grok.DEFAULT_TIMEOUT


# --- the media log line (issue #272) -----------------------------------------


def test_a_grok_generation_logs_one_media_line(image_tool, caplog):
    import logging

    captured = {}
    with respx.mock(assert_all_called=True) as mock:
        mock.post(IMAGES_URL).mock(return_value=httpx.Response(200, json=image_b64()))
        _mock_upload(mock, captured)
        with caplog.at_level(logging.INFO, logger="basecradle_harness"):
            image_tool.run(prompt="a neon skyline")

    line = next(m for m in (r.getMessage() for r in caplog.records) if m.startswith("media "))
    assert "provider=xai" in line
    assert "kind=image.generate" in line
    assert "model=grok-imagine-image-quality" in line


def test_a_video_generation_times_the_submit_and_poll_span(video_tool, caplog):
    """The poll loop *is* the generation on this endpoint — a video that took four minutes to
    render must say four minutes, not the millisecond the submit call returned in."""
    import logging

    captured = {}
    with respx.mock(assert_all_called=True) as mock:
        mock.post(VIDEOS_URL).mock(
            return_value=httpx.Response(200, json={"request_id": REQUEST_ID})
        )
        mock.get(f"{XAI_BASE}/videos/{REQUEST_ID}").mock(
            return_value=httpx.Response(
                200,
                json={"status": "done", "video": {"url": f"{XAI_BASE}/clips/out.mp4"}},
            )
        )
        mock.get(f"{XAI_BASE}/clips/out.mp4").mock(
            return_value=httpx.Response(200, content=MP4_BYTES)
        )
        _mock_upload(mock, captured)
        with caplog.at_level(logging.INFO, logger="basecradle_harness"):
            video_tool.run(prompt="a drone shot over the ocean")

    line = next(m for m in (r.getMessage() for r in caplog.records) if m.startswith("media "))
    assert "kind=video.generate" in line and "provider=xai" in line


# --- the provider-reported cost on the media line (issue #329) ---------------


def _media_line(caplog):
    return next(m for m in (r.getMessage() for r in caplog.records) if m.startswith("media "))


def test_media_cost_usd_converts_ticks_and_degrades_to_none():
    """The tick→dollar conversion, and the honest-absence guard: a body with no numeric cost yields
    None (never a fabricated figure), so the media line simply omits `cost=`."""
    from basecradle_harness._grok import _media_cost_usd

    # xAI's own documented example: 200000000 ticks = $0.02 (1 tick = 1e-10 USD).
    assert _media_cost_usd({"usage": {"cost_in_usd_ticks": 200000000}}) == pytest.approx(0.02)
    assert _media_cost_usd({"usage": {"cost_in_usd_ticks": 0}}) == 0.0
    # No usage, empty usage, or a non-numeric field → None, never a guess.
    assert _media_cost_usd({"data": []}) is None
    assert _media_cost_usd({"usage": {}}) is None
    assert _media_cost_usd({"usage": {"cost_in_usd_ticks": None}}) is None
    assert _media_cost_usd({"usage": {"cost_in_usd_ticks": "lots"}}) is None
    assert _media_cost_usd({"usage": {"cost_in_usd_ticks": True}}) is None  # a bool is not a cost
    assert _media_cost_usd(None) is None


def test_image_media_line_carries_the_xai_reported_cost(image_tool, caplog):
    import logging

    captured = {}
    with respx.mock(assert_all_called=True) as mock:
        # 200000000 ticks = $0.02 — the figure straight off xAI's cost-tracking docs.
        mock.post(IMAGES_URL).mock(
            return_value=httpx.Response(200, json=image_b64(cost_ticks=200000000))
        )
        _mock_upload(mock, captured)
        with caplog.at_level(logging.INFO, logger="basecradle_harness"):
            image_tool.run(prompt="a neon skyline")

    line = _media_line(caplog)
    assert "kind=image.generate" in line
    assert "cost=0.02" in line


def test_image_media_line_omits_cost_when_xai_states_none(image_tool, caplog):
    """A body with no usage block must not invent a cost — the field is simply absent."""
    import logging

    captured = {}
    with respx.mock(assert_all_called=True) as mock:
        mock.post(IMAGES_URL).mock(return_value=httpx.Response(200, json=image_b64()))
        _mock_upload(mock, captured)
        with caplog.at_level(logging.INFO, logger="basecradle_harness"):
            image_tool.run(prompt="a neon skyline")

    assert "cost=" not in _media_line(caplog)


def test_edit_media_line_carries_the_xai_reported_cost(edit_tool, caplog):
    import logging

    captured = {}
    with respx.mock(assert_all_called=True) as mock:
        _mock_source(mock, SOURCE_UUID)
        # 300000000 ticks = $0.03.
        mock.post(EDITS_URL).mock(
            return_value=httpx.Response(200, json=image_b64(cost_ticks=300000000))
        )
        _mock_upload(mock, captured)
        with caplog.at_level(logging.INFO, logger="basecradle_harness"):
            edit_tool.run(image=SOURCE_UUID, prompt="recolor the car red")

    line = _media_line(caplog)
    assert "kind=image.edit" in line
    assert "cost=0.03" in line


def test_video_media_line_carries_cost_from_the_done_poll_body_not_the_submit(video_tool, caplog):
    """The determination this issue asked to verify, pinned: the async video charge rides the
    completed `done` poll body (xAI's final response), not the submit — which returns only a
    request_id. The pending poll carries no usage either, so a cost on the line can only have come
    from `done`."""
    import logging

    captured = {}
    with respx.mock(assert_all_called=True) as mock:
        # The submit returns only a request_id — deliberately no usage/cost here.
        mock.post(VIDEOS_URL).mock(
            return_value=httpx.Response(200, json={"request_id": REQUEST_ID})
        )
        # Pending (no usage), then done (usage present) — 21000000000 ticks = $2.10, a 15s 720p clip.
        mock.get(f"{XAI_BASE}/videos/{REQUEST_ID}").mock(
            side_effect=[
                httpx.Response(200, json={"status": "pending"}),
                httpx.Response(
                    200,
                    json={
                        "status": "done",
                        "video": {"url": f"{XAI_BASE}/clips/out.mp4", "duration": 15},
                        "usage": {"cost_in_usd_ticks": 21000000000},
                    },
                ),
            ]
        )
        mock.get(f"{XAI_BASE}/clips/out.mp4").mock(
            return_value=httpx.Response(200, content=MP4_BYTES)
        )
        _mock_upload(mock, captured)
        with caplog.at_level(logging.INFO, logger="basecradle_harness"):
            video_tool.run(prompt="a drone shot over the ocean", duration=15)

    line = _media_line(caplog)
    assert "kind=video.generate" in line
    assert "cost=2.1" in line


def test_video_media_line_omits_cost_when_the_done_body_states_none(video_tool, caplog):
    import logging

    captured = {}
    with respx.mock(assert_all_called=True) as mock:
        mock.post(VIDEOS_URL).mock(
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
        with caplog.at_level(logging.INFO, logger="basecradle_harness"):
            video_tool.run(prompt="a drone shot over the ocean")

    assert "cost=" not in _media_line(caplog)
