"""The audio-perception tool, against a respx-mocked Audio API and platform.

No live calls: respx stands in for both the OpenAI Audio API (which returns a
transcript) and the BaseCradle SDK transport (asset metadata + the blob fetch).
The fictional cast: Nova Digital (``nova``, AI) listens to a voice note John Doe
(``john``, human) shared on a timeline. The tool runs through a *real* `BaseCradle`
client — only its HTTP is mocked — and a real httpx call to the (fake) Audio API.
"""

import httpx
import pytest
import respx
from basecradle import BaseCradle

from basecradle_harness import HearAudioTool, PlatformContext, PlatformError

BC_URL = "https://basecradle.com"
AUDIO_BASE = "https://api.openai.test/v1"
AUDIO_URL = f"{AUDIO_BASE}/audio/transcriptions"
FAKE_TOKEN = "bc_uat_KqI8zFxkQ0OZ8vYwT7mWcVtR3nSdLpEa"
FAKE_KEY = "sk-test-0123456789abcdefghijklmnop"

NOVA_UUID = "019e7750-66ee-79c8-ad8a-bbb6ea7c2bcc"
JOHN_UUID = "019e7750-66ee-7e50-9e54-3bf8c3d6a8f1"
TIMELINE_UUID = "019e7750-66ee-7f53-829f-13a8a710b6da"
A_AUDIO = "019e7754-7d4e-7f50-8162-4d5e6f708192"

BLOB_URL = f"{BC_URL}/rails/active_storage/blobs/redirect/aud789/voice.mp3"
MP3_BYTES = b"ID3\x04\x00 fake mp3 frames"
TRANSCRIPT = "Hey Nova, can you take a look at the incident timeline?"


def audio_asset(*, content_type="audio/mpeg", byte_size=None, filename="voice.mp3"):
    return {
        "type": "asset",
        "created_at": "2026-06-04T00:00:00.000Z",
        "user": {"uuid": JOHN_UUID, "handle": "john", "name": "John Doe", "kind": "human"},
        "timeline": {"uuid": TIMELINE_UUID},
        "content": {
            "uuid": A_AUDIO,
            "description": "A voice note",
            "file": {
                "filename": filename,
                "byte_size": byte_size if byte_size is not None else len(MP3_BYTES),
                "content_type": content_type,
                "checksum": "Yp9p9C8m6Xv2qS1nKQ0r3w==",
                "url": BLOB_URL,
            },
        },
    }


@pytest.fixture
def client():
    c = BaseCradle(token=FAKE_TOKEN)
    yield c
    c.close()


@pytest.fixture
def tool(client):
    """A HearAudioTool bound to John's timeline, pointed at the fake Audio API."""
    t = HearAudioTool(api_key=FAKE_KEY, base_url=AUDIO_BASE)
    t.bind(PlatformContext(client=client, timeline=TIMELINE_UUID))
    return t


# --- the happy path ----------------------------------------------------------


def test_listen_transcribes_the_audio(tool):
    captured = {}

    def transcribe(request):
        captured["body"] = request.content
        captured["content_type"] = request.headers["content-type"]
        return httpx.Response(200, json={"text": TRANSCRIPT})

    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/assets/{A_AUDIO}").mock(
            return_value=httpx.Response(200, json={"asset": audio_asset()})
        )
        mock.get(BLOB_URL).mock(return_value=httpx.Response(200, content=MP3_BYTES))
        mock.post(AUDIO_URL).mock(side_effect=transcribe)
        result = tool.run(uuid=A_AUDIO)

    # The audio rode as a multipart upload carrying the blob bytes and the model name.
    assert "multipart/form-data" in captured["content_type"]
    assert MP3_BYTES in captured["body"]
    assert b"gpt-4o-transcribe" in captured["body"]
    # ...and the transcript comes back for the model to read.
    assert TRANSCRIPT in result
    assert "Transcript:" in result
    assert A_AUDIO in result  # the asset metadata line


def test_listen_passes_the_api_key_and_filename(tool):
    def transcribe(request):
        # The filename is what the Audio API uses to detect the format.
        assert b'filename="voice.mp3"' in request.content
        assert request.headers["Authorization"] == f"Bearer {FAKE_KEY}"
        return httpx.Response(200, json={"text": TRANSCRIPT})

    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/assets/{A_AUDIO}").mock(
            return_value=httpx.Response(200, json={"asset": audio_asset()})
        )
        mock.get(BLOB_URL).mock(return_value=httpx.Response(200, content=MP3_BYTES))
        mock.post(AUDIO_URL).mock(side_effect=transcribe)
        tool.run(uuid=A_AUDIO)


def test_api_key_falls_back_to_env(client, monkeypatch):
    monkeypatch.setenv("AI_API_KEY", FAKE_KEY)
    t = HearAudioTool(base_url=AUDIO_BASE)  # no key passed
    t.bind(PlatformContext(client=client, timeline=TIMELINE_UUID))
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/assets/{A_AUDIO}").mock(
            return_value=httpx.Response(200, json={"asset": audio_asset()})
        )
        mock.get(BLOB_URL).mock(return_value=httpx.Response(200, content=MP3_BYTES))
        post = mock.post(AUDIO_URL).mock(
            return_value=httpx.Response(200, json={"text": TRANSCRIPT})
        )
        t.run(uuid=A_AUDIO)
    assert post.calls.last.request.headers["Authorization"] == f"Bearer {FAKE_KEY}"


# --- the wrong file comes back as a clean note, not a failure ----------------


def test_listen_rejects_a_non_audio_file(tool):
    with respx.mock(assert_all_called=True) as mock:
        # No blob route, no Audio route: a non-audio asset is refused without
        # downloading or calling the provider.
        mock.get(f"{BC_URL}/assets/{A_AUDIO}").mock(
            return_value=httpx.Response(
                200, json={"asset": audio_asset(content_type="image/png", filename="cat.png")}
            )
        )
        result = tool.run(uuid=A_AUDIO)

    assert "not an audio file" in result
    assert "Transcript" not in result


def test_listen_rejects_an_empty_file(tool):
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/assets/{A_AUDIO}").mock(
            return_value=httpx.Response(200, json={"asset": audio_asset(byte_size=0)})
        )
        result = tool.run(uuid=A_AUDIO)

    assert "empty file" in result


def test_listen_rejects_an_oversized_file(tool):
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/assets/{A_AUDIO}").mock(
            return_value=httpx.Response(
                200, json={"asset": audio_asset(byte_size=30 * 1024 * 1024)}
            )
        )
        result = tool.run(uuid=A_AUDIO)

    assert "too large to listen to" in result


def test_no_speech_detected_is_a_clean_note(tool):
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/assets/{A_AUDIO}").mock(
            return_value=httpx.Response(200, json={"asset": audio_asset()})
        )
        mock.get(BLOB_URL).mock(return_value=httpx.Response(200, content=MP3_BYTES))
        mock.post(AUDIO_URL).mock(return_value=httpx.Response(200, json={"text": "   "}))
        result = tool.run(uuid=A_AUDIO)

    assert "no speech was detected" in result


# --- failures come back as model-readable text -------------------------------


def test_missing_uuid_is_a_friendly_error(tool):
    assert "needs the audio asset's uuid" in tool.run(uuid="   ")


def test_no_api_key_is_a_friendly_error(client, monkeypatch):
    monkeypatch.delenv("AI_API_KEY", raising=False)
    t = HearAudioTool(base_url=AUDIO_BASE)  # no key passed, none in env
    t.bind(PlatformContext(client=client, timeline=TIMELINE_UUID))
    with respx.mock(assert_all_called=True) as mock:
        # The asset is fetched (to learn it's audio); the provider is never reached.
        mock.get(f"{BC_URL}/assets/{A_AUDIO}").mock(
            return_value=httpx.Response(200, json={"asset": audio_asset()})
        )
        result = t.run(uuid=A_AUDIO)
    assert "no API key" in result


def test_an_api_error_is_relayed_to_the_model(tool):
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/assets/{A_AUDIO}").mock(
            return_value=httpx.Response(200, json={"asset": audio_asset()})
        )
        mock.get(BLOB_URL).mock(return_value=httpx.Response(200, content=MP3_BYTES))
        mock.post(AUDIO_URL).mock(return_value=httpx.Response(400, text="unsupported format"))
        result = tool.run(uuid=A_AUDIO)

    assert "Error transcribing audio" in result


def test_a_missing_text_field_is_a_friendly_error(tool):
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/assets/{A_AUDIO}").mock(
            return_value=httpx.Response(200, json={"asset": audio_asset()})
        )
        mock.get(BLOB_URL).mock(return_value=httpx.Response(200, content=MP3_BYTES))
        mock.post(AUDIO_URL).mock(return_value=httpx.Response(200, json={}))
        result = tool.run(uuid=A_AUDIO)

    assert "no text" in result


def test_a_non_object_body_is_a_friendly_error(tool):
    """A malformed body (a bare JSON string, e.g. from a proxy) relays cleanly, not a crash."""
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/assets/{A_AUDIO}").mock(
            return_value=httpx.Response(200, json={"asset": audio_asset()})
        )
        mock.get(BLOB_URL).mock(return_value=httpx.Response(200, content=MP3_BYTES))
        mock.post(AUDIO_URL).mock(return_value=httpx.Response(200, json="rate limited"))
        result = tool.run(uuid=A_AUDIO)

    assert "Error transcribing audio" in result
    assert "no text" in result


def test_a_transport_failure_is_relayed_to_the_model(tool):
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/assets/{A_AUDIO}").mock(
            return_value=httpx.Response(200, json={"asset": audio_asset()})
        )
        mock.get(BLOB_URL).mock(return_value=httpx.Response(200, content=MP3_BYTES))
        mock.post(AUDIO_URL).mock(side_effect=httpx.ConnectError("no route"))
        result = tool.run(uuid=A_AUDIO)

    assert "could not reach the transcription API" in result


# --- the tool contract & binding ---------------------------------------------


def test_spec_requires_a_uuid(tool):
    spec = tool.to_spec()
    assert spec.name == "listen"
    assert spec.parameters["required"] == ["uuid"]


def test_an_unbound_tool_raises_platform_error(monkeypatch):
    monkeypatch.setenv("AI_API_KEY", FAKE_KEY)
    with pytest.raises(PlatformError):
        # Even fetching the asset needs a bound platform context.
        HearAudioTool(api_key=FAKE_KEY, base_url=AUDIO_BASE).run(uuid=A_AUDIO)


def test_listen_loads_under_the_locked_profile(client):
    """Audio perception is platform I/O, permitted by the safe default profile."""
    from basecradle_harness import Policy, ToolRegistry

    registry = ToolRegistry(policy=Policy.locked())
    registry.register(HearAudioTool())
    assert "listen" in registry


# --- the media log line (issue #272) -----------------------------------------


def test_a_transcription_logs_one_media_line(tool, caplog):
    import logging

    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/assets/{A_AUDIO}").mock(
            return_value=httpx.Response(200, json={"asset": audio_asset()})
        )
        mock.get(BLOB_URL).mock(return_value=httpx.Response(200, content=MP3_BYTES))
        mock.post(AUDIO_URL).mock(return_value=httpx.Response(200, json={"text": TRANSCRIPT}))
        with caplog.at_level(logging.INFO, logger="basecradle_harness"):
            tool.run(uuid=A_AUDIO)

    line = next(m for m in (r.getMessage() for r in caplog.records) if m.startswith("media "))
    assert "provider=openai" in line and "kind=audio.transcribe" in line
