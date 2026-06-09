"""The image-generation tool, against a respx-mocked Images API and platform.

No live calls: respx stands in for both the OpenAI Images API (which returns a
base64 image) and the BaseCradle SDK transport (which takes the upload). The
fictional cast: Nova Digital (``nova``, AI) generates an image onto John Doe's
timeline. The tool is exercised through a *real* `BaseCradle` client — only its
HTTP is mocked — and a real httpx call to the (fake) Images endpoint.
"""

import base64
import json

import httpx
import pytest
import respx
from basecradle import BaseCradle

from basecradle_harness import GenerateImageTool, PlatformContext, PlatformError

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


# --- failures come back as model-readable text -------------------------------


def test_missing_prompt_is_a_friendly_error(tool):
    assert "needs a 'prompt'" in tool.run(prompt="   ")


def test_no_api_key_is_a_friendly_error(client, monkeypatch):
    monkeypatch.delenv("AI_PROVIDER_API_KEY", raising=False)
    t = GenerateImageTool(base_url=IMAGES_BASE)  # no key passed, none in env
    t.bind(PlatformContext(client=client, timeline=TIMELINE_UUID))
    assert "no API key" in t.run(prompt="a cat")


def test_api_key_falls_back_to_env(client, monkeypatch):
    monkeypatch.setenv("AI_PROVIDER_API_KEY", FAKE_KEY)
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


# --- binding -----------------------------------------------------------------


def test_an_unbound_tool_raises_platform_error(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER_API_KEY", FAKE_KEY)
    with respx.mock(assert_all_called=True) as mock:
        mock.post(IMAGES_URL).mock(return_value=httpx.Response(200, json=images_response()))
        with pytest.raises(PlatformError):
            # Generation succeeds, but the upload needs a bound platform context.
            GenerateImageTool(api_key=FAKE_KEY, base_url=IMAGES_BASE).run(prompt="a cat")
