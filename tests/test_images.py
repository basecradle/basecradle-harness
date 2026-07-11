"""The image tools (generate + edit), against a respx-mocked Images API and platform.

No live calls: respx stands in for both the OpenAI Images API (which returns a
base64 image) and the BaseCradle SDK transport (which takes the upload). The
fictional cast: Nova Digital (``nova``, AI) generates/edits an image onto John Doe's
timeline. The tools are exercised through a *real* `BaseCradle` client — only its
HTTP is mocked — and a real httpx call to the (fake) Images endpoint.

These tests assert the harness's half of the contract (the params it sends, the
filename extension it posts). The ground-truth checks the handoff calls for — the
posted Asset's actual pixels / content-type / file magic — are the capital's live
@jt verification, which cannot run offline against a mock.
"""

import base64
import json

import httpx
import pytest
import respx
from basecradle import BaseCradle

from basecradle_harness import (
    EditImageTool,
    GenerateImageTool,
    PlatformContext,
    PlatformError,
)

BC_URL = "https://basecradle.com"
IMAGES_BASE = "https://api.openai.test/v1"
IMAGES_URL = f"{IMAGES_BASE}/images/generations"
FAKE_TOKEN = "bc_uat_KqI8zFxkQ0OZ8vYwT7mWcVtR3nSdLpEa"
FAKE_KEY = "sk-test-0123456789abcdefghijklmnop"

NOVA_UUID = "019e7750-66ee-79c8-ad8a-bbb6ea7c2bcc"
JOHN_UUID = "019e7750-66ee-7e50-9e54-3bf8c3d6a8f1"
TIMELINE_UUID = "019e7750-66ee-7f53-829f-13a8a710b6da"
A_IMG = "019e7754-7d4e-7f50-8162-4d5e6f708192"

PNG_BYTES = b"\x89PNG\r\n\x1a\n generated pixels"


def images_response(data=PNG_BYTES):
    return {"data": [{"b64_json": base64.b64encode(data).decode("ascii")}]}


def asset_response(*, filename="a-red-cube.png"):
    return {
        "asset": {
            "type": "asset",
            "created_at": "2026-06-04T00:00:00.000Z",
            "user": {"uuid": NOVA_UUID, "handle": "nova", "name": "Nova Digital", "kind": "ai"},
            "timeline": {"uuid": TIMELINE_UUID},
            "content": {
                "uuid": A_IMG,
                "description": "Generated image",
                "file": {
                    "filename": filename,
                    "byte_size": len(PNG_BYTES),
                    "content_type": "image/png",
                    "checksum": "Yp9p9C8m6Xv2qS1nKQ0r3w==",
                    "url": f"{BC_URL}/blobs/{A_IMG}",
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


@pytest.fixture
def client():
    c = BaseCradle(token=FAKE_TOKEN)
    yield c
    c.close()


@pytest.fixture
def tool(client):
    """A GenerateImageTool bound to John's timeline, pointed at the fake Images API."""
    t = GenerateImageTool(api_key=FAKE_KEY, base_url=IMAGES_BASE)
    t.bind(PlatformContext(client=client, timeline=TIMELINE_UUID))
    return t


# --- the happy path ----------------------------------------------------------


def test_generate_posts_the_image_as_an_asset(tool):
    captured = {}

    def upload(request):
        captured["body"] = request.content
        return httpx.Response(201, json=asset_response())

    with respx.mock(assert_all_called=True) as mock:
        gen = mock.post(IMAGES_URL).mock(return_value=httpx.Response(200, json=images_response()))
        mock.get(f"{BC_URL}/timelines/{TIMELINE_UUID}").mock(
            return_value=httpx.Response(200, json=_timeline_envelope())
        )
        mock.post(f"{BC_URL}/timelines/{TIMELINE_UUID}/assets").mock(side_effect=upload)
        result = tool.run(prompt="a red cube on a white table")

    # The model named the image model and prompt to the Images API...
    sent = json.loads(gen.calls.last.request.content)
    assert sent["model"] == "gpt-image-2"
    assert sent["prompt"] == "a red cube on a white table"
    # ...the decoded bytes were uploaded as multipart...
    assert PNG_BYTES in captured["body"]
    # ...and the tool reports the new asset (with its uuid) for the model to read.
    assert "Generated and posted" in result
    assert A_IMG in result


def test_generate_derives_a_filename_from_the_prompt(tool):
    captured = {}

    def upload(request):
        captured["body"] = request.content
        return httpx.Response(201, json=asset_response())

    with respx.mock(assert_all_called=True) as mock:
        mock.post(IMAGES_URL).mock(return_value=httpx.Response(200, json=images_response()))
        mock.get(f"{BC_URL}/timelines/{TIMELINE_UUID}").mock(
            return_value=httpx.Response(200, json=_timeline_envelope())
        )
        mock.post(f"{BC_URL}/timelines/{TIMELINE_UUID}/assets").mock(side_effect=upload)
        tool.run(prompt="A Red Cube!")

    # Slugified from the prompt, lowercased, hyphenated, .png extension.
    assert b"a-red-cube.png" in captured["body"]


def test_generate_honors_an_explicit_size(tool):
    with respx.mock(assert_all_called=True) as mock:
        gen = mock.post(IMAGES_URL).mock(return_value=httpx.Response(200, json=images_response()))
        mock.get(f"{BC_URL}/timelines/{TIMELINE_UUID}").mock(
            return_value=httpx.Response(200, json=_timeline_envelope())
        )
        mock.post(f"{BC_URL}/timelines/{TIMELINE_UUID}/assets").mock(
            return_value=httpx.Response(201, json=asset_response())
        )
        tool.run(prompt="a landscape", size="1536x1024")

    assert json.loads(gen.calls.last.request.content)["size"] == "1536x1024"


# --- Part A: full gpt-image-2 coverage on generate ---------------------------


def _mock_generate(mock, captured):
    """Wire the Images generate endpoint + the upload, capturing the uploaded body."""

    def upload(request):
        captured["upload"] = request.content
        return httpx.Response(201, json=asset_response())

    gen = mock.post(IMAGES_URL).mock(return_value=httpx.Response(200, json=images_response()))
    mock.get(f"{BC_URL}/timelines/{TIMELINE_UUID}").mock(
        return_value=httpx.Response(200, json=_timeline_envelope())
    )
    mock.post(f"{BC_URL}/timelines/{TIMELINE_UUID}/assets").mock(side_effect=upload)
    return gen


@pytest.mark.parametrize(
    "output_format,extension",
    [("png", b".png"), ("jpeg", b".jpg"), ("webp", b".webp")],
)
def test_generate_output_format_is_passed_and_drives_the_filename(tool, output_format, extension):
    captured = {}
    with respx.mock(assert_all_called=True) as mock:
        gen = _mock_generate(mock, captured)
        tool.run(prompt="a cat", output_format=output_format)

    # The format reaches the Images API...
    assert json.loads(gen.calls.last.request.content)["output_format"] == output_format
    # ...and the posted file's extension follows it (so its content-type does too).
    assert extension in captured["upload"]


def test_generate_passes_quality_background_and_compression(tool):
    captured = {}
    with respx.mock(assert_all_called=True) as mock:
        gen = _mock_generate(mock, captured)
        tool.run(
            prompt="a cat",
            quality="high",
            background="opaque",
            output_format="jpeg",
            output_compression=40,
        )

    sent = json.loads(gen.calls.last.request.content)
    assert sent["quality"] == "high"
    assert sent["background"] == "opaque"
    assert sent["output_compression"] == 40


def test_generate_drops_compression_for_png(tool):
    # output_compression is jpeg/webp-only — OpenAI hard-400s it on png. The model fills
    # the schema field in freely, so the tool must drop it for png or every such png
    # request fails. (Capital live-verify bug 1, #140.)
    captured = {}
    with respx.mock(assert_all_called=True) as mock:
        gen = _mock_generate(mock, captured)
        tool.run(prompt="a cat", output_format="png", output_compression=40)

    assert "output_compression" not in json.loads(gen.calls.last.request.content)


def test_generate_drops_compression_when_format_unset(tool):
    # png is the default when no format is named, so an unset format must drop it too.
    captured = {}
    with respx.mock(assert_all_called=True) as mock:
        gen = _mock_generate(mock, captured)
        tool.run(prompt="a cat", output_compression=40)

    assert "output_compression" not in json.loads(gen.calls.last.request.content)


def test_generate_omits_unset_coverage_params(tool):
    # An unset knob is left out entirely, so the API picks its own default rather than
    # the harness inventing one. Only `size` (which has a tool default) always carries.
    captured = {}
    with respx.mock(assert_all_called=True) as mock:
        gen = _mock_generate(mock, captured)
        tool.run(prompt="a cat")

    sent = json.loads(gen.calls.last.request.content)
    assert "size" in sent
    for absent in ("quality", "background", "output_format", "output_compression"):
        assert absent not in sent


# --- failures come back as model-readable text -------------------------------


def test_missing_prompt_is_a_friendly_error(tool):
    assert "needs a 'prompt'" in tool.run(prompt="   ")


def test_no_api_key_is_a_friendly_error(client, monkeypatch):
    monkeypatch.delenv("AI_API_KEY", raising=False)
    t = GenerateImageTool(base_url=IMAGES_BASE)  # no key passed, none in env
    t.bind(PlatformContext(client=client, timeline=TIMELINE_UUID))
    assert "no API key" in t.run(prompt="a cat")


def test_api_key_falls_back_to_env(client, monkeypatch):
    monkeypatch.setenv("AI_API_KEY", FAKE_KEY)
    t = GenerateImageTool(base_url=IMAGES_BASE)
    t.bind(PlatformContext(client=client, timeline=TIMELINE_UUID))
    with respx.mock(assert_all_called=True) as mock:
        gen = mock.post(IMAGES_URL).mock(return_value=httpx.Response(200, json=images_response()))
        mock.get(f"{BC_URL}/timelines/{TIMELINE_UUID}").mock(
            return_value=httpx.Response(200, json=_timeline_envelope())
        )
        mock.post(f"{BC_URL}/timelines/{TIMELINE_UUID}/assets").mock(
            return_value=httpx.Response(201, json=asset_response())
        )
        t.run(prompt="a cat")
    assert gen.calls.last.request.headers["Authorization"] == f"Bearer {FAKE_KEY}"


def test_an_api_error_is_relayed_to_the_model(tool):
    with respx.mock(assert_all_called=True) as mock:
        mock.post(IMAGES_URL).mock(
            return_value=httpx.Response(400, text="content policy violation")
        )
        result = tool.run(prompt="something disallowed")

    assert "Error generating image" in result
    # The provider's body — not a generic status — reaches the model.
    assert "content policy violation" in result


def test_an_api_error_relays_the_openai_message_not_a_bare_status(tool):
    # The real reason lives in OpenAI's JSON body under error.message; the exception's
    # own str is only "Provider returned HTTP 400". Surface the real reason so the AI
    # can relay the true cause to the user. (Capital live-verify bug 2 / Principle 5.)
    body = {
        "error": {
            "message": "Compression less than 100 is not supported for PNG output format",
            "code": "invalid_png_output_compression",
        }
    }
    with respx.mock(assert_all_called=True) as mock:
        mock.post(IMAGES_URL).mock(return_value=httpx.Response(400, json=body))
        result = tool.run(prompt="a cat")

    assert "Compression less than 100 is not supported for PNG output format" in result
    assert "Provider returned HTTP" not in result


def test_an_empty_data_array_is_a_friendly_error(tool):
    with respx.mock(assert_all_called=True) as mock:
        mock.post(IMAGES_URL).mock(return_value=httpx.Response(200, json={"data": []}))
        result = tool.run(prompt="a cat")

    assert "no image data" in result


def test_a_transport_failure_is_relayed_to_the_model(tool):
    with respx.mock(assert_all_called=True) as mock:
        mock.post(IMAGES_URL).mock(side_effect=httpx.ConnectError("no route"))
        result = tool.run(prompt="a cat")

    assert "could not reach the image API" in result


# --- the request timeout (issue #219) ----------------------------------------


def test_default_timeout_clears_the_measured_high_quality_latency():
    # A `gpt-image-2` `quality: high` edit was measured at ~133s live; agents pick high
    # naturally for fidelity work, so the ceiling must clear it with headroom (issue #219).
    from basecradle_harness._images import DEFAULT_TIMEOUT

    assert DEFAULT_TIMEOUT >= 133.0
    assert DEFAULT_TIMEOUT == 300.0


def test_the_timeout_reaches_the_openai_client(client, monkeypatch):
    # Whatever timeout the tool carries is the one handed to the SDK client, so the ceiling
    # bump actually takes effect at call time rather than being decorative.
    import basecradle_harness._images as images

    captured = {}

    class _FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class _FakeOpenAI:
        OpenAI = _FakeClient

    monkeypatch.setattr(images, "require_openai_sdk", lambda: _FakeOpenAI)
    t = GenerateImageTool(api_key=FAKE_KEY, base_url=IMAGES_BASE)
    t._client(FAKE_KEY)
    assert captured["timeout"] == images.DEFAULT_TIMEOUT


# --- binding -----------------------------------------------------------------


def test_an_unbound_tool_raises_platform_error(monkeypatch):
    monkeypatch.setenv("AI_API_KEY", FAKE_KEY)
    with respx.mock(assert_all_called=True) as mock:
        mock.post(IMAGES_URL).mock(return_value=httpx.Response(200, json=images_response()))
        with pytest.raises(PlatformError):
            # Generation succeeds, but the upload needs a bound platform context.
            GenerateImageTool(api_key=FAKE_KEY, base_url=IMAGES_BASE).run(prompt="a cat")


# =============================================================================
# Part B: edit_image — /v1/images/edits, source bytes (not URLs), masks, multi.
# =============================================================================

EDITS_URL = f"{IMAGES_BASE}/images/edits"

SOURCE_UUID = "019e7760-0000-7000-8000-000000000001"
SOURCE2_UUID = "019e7760-0000-7000-8000-000000000002"
MASK_UUID = "019e7760-0000-7000-8000-000000000003"
SOURCE_BYTES = b"\x89PNG\r\n\x1a\n source one pixels"
SOURCE2_BYTES = b"\x89PNG\r\n\x1a\n source two pixels"
MASK_BYTES = b"\x89PNG\r\n\x1a\n alpha mask pixels"


def source_asset_response(uuid, *, filename="source.png", content_type="image/png"):
    """An `assets.get(uuid)` envelope for a source/mask image to be edited."""
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
                    "filename": filename,
                    "byte_size": 32,
                    "content_type": content_type,
                    "checksum": "Yp9p9C8m6Xv2qS1nKQ0r3w==",
                    "url": f"{BC_URL}/blobs/{uuid}",
                },
            },
        }
    }


@pytest.fixture
def edit_tool(client):
    """An EditImageTool bound to John's timeline, pointed at the fake Images API."""
    t = EditImageTool(api_key=FAKE_KEY, base_url=IMAGES_BASE)
    t.bind(PlatformContext(client=client, timeline=TIMELINE_UUID))
    return t


def _mock_source(mock, uuid, data):
    """Wire `assets.get(uuid)` and its blob download to return `data`."""
    mock.get(f"{BC_URL}/assets/{uuid}").mock(
        return_value=httpx.Response(200, json=source_asset_response(uuid))
    )
    mock.get(f"{BC_URL}/blobs/{uuid}").mock(return_value=httpx.Response(200, content=data))


def _mock_edit_upload(mock, captured):
    """Wire the Images edit endpoint + the result upload, capturing both bodies."""

    def edit(request):
        captured["edit"] = request.content
        return httpx.Response(200, json=images_response())

    def upload(request):
        captured["upload"] = request.content
        return httpx.Response(201, json=asset_response())

    mock.post(EDITS_URL).mock(side_effect=edit)
    mock.get(f"{BC_URL}/timelines/{TIMELINE_UUID}").mock(
        return_value=httpx.Response(200, json=_timeline_envelope())
    )
    mock.post(f"{BC_URL}/timelines/{TIMELINE_UUID}/assets").mock(side_effect=upload)


def test_edit_sends_source_bytes_not_a_url_and_posts_the_result(edit_tool):
    captured = {}
    with respx.mock(assert_all_called=True) as mock:
        _mock_source(mock, SOURCE_UUID, SOURCE_BYTES)
        _mock_edit_upload(mock, captured)
        result = edit_tool.run(image=[SOURCE_UUID], prompt="recolor the car red")

    body = captured["edit"]
    # The source's *bytes* ride the multipart body (the endpoint rejects URLs)...
    assert SOURCE_BYTES in body
    assert b'name="image[]"' in body
    # ...the prompt and model are sent as form fields...
    assert b"recolor the car red" in body
    assert b"gpt-image-2" in body
    # ...and the edited result is posted as a new asset the model can read.
    assert "Edited and posted" in result
    assert A_IMG in result


def test_edit_accepts_a_bare_string_for_a_single_source(edit_tool):
    captured = {}
    with respx.mock(assert_all_called=True) as mock:
        _mock_source(mock, SOURCE_UUID, SOURCE_BYTES)
        _mock_edit_upload(mock, captured)
        # A model may pass a single uuid as a string rather than a one-element list.
        result = edit_tool.run(image=SOURCE_UUID, prompt="make it night")

    assert SOURCE_BYTES in captured["edit"]
    assert "Edited and posted" in result


def test_edit_composites_multiple_sources(edit_tool):
    captured = {}
    with respx.mock(assert_all_called=True) as mock:
        _mock_source(mock, SOURCE_UUID, SOURCE_BYTES)
        _mock_source(mock, SOURCE2_UUID, SOURCE2_BYTES)
        _mock_edit_upload(mock, captured)
        edit_tool.run(image=[SOURCE_UUID, SOURCE2_UUID], prompt="combine these")

    body = captured["edit"]
    assert SOURCE_BYTES in body and SOURCE2_BYTES in body
    assert body.count(b'name="image[]"') == 2


def test_edit_with_a_mask_includes_the_mask_part(edit_tool):
    captured = {}
    with respx.mock(assert_all_called=True) as mock:
        _mock_source(mock, SOURCE_UUID, SOURCE_BYTES)
        _mock_source(mock, MASK_UUID, MASK_BYTES)
        _mock_edit_upload(mock, captured)
        edit_tool.run(image=[SOURCE_UUID], prompt="repaint the masked area", mask=MASK_UUID)

    body = captured["edit"]
    assert b'name="mask"' in body
    assert MASK_BYTES in body


def test_edit_output_format_is_passed_and_drives_the_filename(edit_tool):
    captured = {}
    with respx.mock(assert_all_called=True) as mock:
        _mock_source(mock, SOURCE_UUID, SOURCE_BYTES)
        _mock_edit_upload(mock, captured)
        edit_tool.run(image=[SOURCE_UUID], prompt="recolor", output_format="jpeg")

    assert b'name="output_format"' in captured["edit"]
    assert b"jpeg" in captured["edit"]
    assert b".jpg" in captured["upload"]  # posted file's extension follows the format


# --- edit failures come back as model-readable text --------------------------


def test_edit_without_a_prompt_is_a_friendly_error(edit_tool):
    assert "needs a 'prompt'" in edit_tool.run(image=[SOURCE_UUID], prompt="   ")


def test_edit_without_a_source_image_is_a_friendly_error(edit_tool):
    assert "needs at least one source" in edit_tool.run(image=[], prompt="recolor")
    assert "needs at least one source" in edit_tool.run(image=None, prompt="recolor")


def test_edit_with_no_api_key_is_a_friendly_error(client, monkeypatch):
    monkeypatch.delenv("AI_API_KEY", raising=False)
    t = EditImageTool(base_url=IMAGES_BASE)  # no key passed, none in env
    t.bind(PlatformContext(client=client, timeline=TIMELINE_UUID))
    assert "no API key" in t.run(image=[SOURCE_UUID], prompt="recolor")


def test_edit_with_a_bad_source_uuid_is_relayed_to_the_model(edit_tool):
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/assets/{SOURCE_UUID}").mock(
            return_value=httpx.Response(404, json={"detail": "asset not found"})
        )
        result = edit_tool.run(image=[SOURCE_UUID], prompt="recolor")

    # The SDK's BaseCradleError is caught and relayed as model-readable text (naming
    # the offending uuid), never raised as a traceback.
    assert "couldn't read asset" in result
    assert SOURCE_UUID in result


def test_edit_relays_an_images_api_error(edit_tool):
    with respx.mock(assert_all_called=True) as mock:
        _mock_source(mock, SOURCE_UUID, SOURCE_BYTES)
        mock.post(EDITS_URL).mock(return_value=httpx.Response(400, text="bad mask"))
        result = edit_tool.run(image=[SOURCE_UUID], prompt="recolor")

    assert "Error editing image" in result
    assert "bad mask" in result  # the provider's reason reaches the model, not "HTTP 400"


# --- the media log line (issue #272) -----------------------------------------


def test_a_generation_logs_one_media_line(tool, caplog):
    """Image work is slow and expensive, so the journal names the vendor, the kind, and the
    time the *generation* took (not the Asset upload that follows it)."""
    import logging

    with respx.mock(assert_all_called=True) as mock:
        mock.post(IMAGES_URL).mock(return_value=httpx.Response(200, json=images_response()))
        mock.get(f"{BC_URL}/timelines/{TIMELINE_UUID}").mock(
            return_value=httpx.Response(200, json=_timeline_envelope())
        )
        mock.post(f"{BC_URL}/timelines/{TIMELINE_UUID}/assets").mock(
            return_value=httpx.Response(201, json=asset_response())
        )
        with caplog.at_level(logging.INFO, logger="basecradle_harness"):
            tool.run(prompt="a red cube on a white table")

    line = next(m for m in (r.getMessage() for r in caplog.records) if m.startswith("media "))
    assert "provider=openai" in line
    assert "kind=image.generate" in line
    assert "model=gpt-image-2" in line
