"""Let the agent hear: fetch an audio asset and transcribe what was said.

The audio analog of the assets tool's `view` (which lets a peer *see* an image):
`HearAudioTool` lets a peer *listen* to one. On a platform that carries audio — TTS,
music, voice notes — a peer that can't hear is half-deaf; this closes that gap by
turning an audio asset into text the model can read and reason over.

Why a sibling tool, not an action on the assets tool
----------------------------------------------------
`view` is pure platform I/O: it downloads the image and hands the bytes to the model,
which sees the picture natively — no model-provider call. Transcription is different:
it needs a *provider* call (OpenAI's transcription endpoint), exactly like image
*generation*. So this follows `GenerateImageTool`'s shape, not `view`'s — a separate
`PlatformTool` that holds the agent's ``AI_API_KEY`` and reaches OpenAI's Audio endpoint
**through the ``openai`` SDK** (``client.audio.transcriptions``), never hand-rolled HTTP —
the same vendor-SDK rule the model loop follows (issue #158), keeping the brain/body boundary
clean (the platform SDK never reaches the model provider, and vice versa). The capability is
one small tool class either way.

It mirrors `view`'s on-demand, ephemeral shape: the agent listens only when it
chooses (never eagerly inlined), a non-audio asset comes back as a clean note rather
than a failure, and an oversized one is described, not force-fed. The transcription
model is OpenAI's Audio API, sharing the agent's one key (``gpt-5.4-mini`` reasons,
``gpt-image-2`` paints, ``gpt-4o-transcribe`` listens).

Video is deliberately out of scope (heavier, and frame extraction would collide with
the no-subprocess safety boundary) — when it comes, it gets its own pure-Python path.
"""

from __future__ import annotations

import os

from basecradle_harness._assets import _describe, _download, _is_audio
from basecradle_harness._exceptions import ProviderConnectionError, ProviderError
from basecradle_harness._openai import require_openai_sdk, sdk_error_context
from basecradle_harness._platform import PlatformTool

#: OpenAI's Audio API root. Transcription is an OpenAI service; this changes only for
#: a proxy, not to reach another vendor (the key is the OpenAI key).
DEFAULT_BASE_URL = "https://api.openai.com/v1"
#: The transcription model. ``gpt-4o-transcribe`` is the current speech-to-text model;
#: ``whisper-1`` is the older alternative the same endpoint accepts.
DEFAULT_MODEL = "gpt-4o-transcribe"
#: Transcription is slow next to a chat call — give it room before giving up.
DEFAULT_TIMEOUT = 120.0
#: The largest audio file to transcribe. 25 MiB is OpenAI's per-file upload ceiling;
#: larger than this is described, not sent (it would only be rejected downstream).
MAX_AUDIO_BYTES = 25 * 1024 * 1024


class HearAudioTool(PlatformTool):
    """Transcribe an audio asset on a timeline so the agent can read what was said.

    A `PlatformTool`: it fetches the asset through the bound SDK client, so the
    hosting agent (`TimelineAgent`/`WakeAgent`) binds it before the loop, exactly
    like the assets tool. The transcription model's key is the agent's
    ``AI_API_KEY`` unless an explicit `api_key` is passed.

    Args:
        api_key: The OpenAI key for the Audio API. Falls back to
            ``AI_API_KEY`` at call time, so constructing the tool needs no
            secret (a keyless construction just errors, readably, if used).
        base_url: The Audio API root. Defaults to OpenAI.
        model: The transcription model. Defaults to ``gpt-4o-transcribe``.
        timeout: Per-request timeout in seconds (transcription is slow).
    """

    name = "listen"
    description = (
        "Listen to an audio file on the timeline and read what was said. Give the "
        "audio asset's uuid (find it with the assets tool's 'list'); the audio is "
        "fetched and transcribed, and the transcript is returned for you to read and "
        "reason over — the way 'view' lets you see an image. A non-audio file comes "
        "back with a clean note, not an error. Use this for voice notes, TTS, or any "
        "spoken-word audio a peer shares."
    )
    parameters = {
        "type": "object",
        "properties": {
            "uuid": {
                "type": "string",
                "description": "The audio asset's uuid. Get it from the assets tool's 'list'.",
            },
        },
        "required": ["uuid"],
    }

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout

    def run(self, uuid: str | None = None) -> str:
        """Fetch the audio asset, transcribe it, and return the transcript to read."""
        if not uuid or not uuid.strip():
            return "Error: 'listen' needs the audio asset's uuid. Use the assets tool's 'list'."

        asset = self.context.client.assets.get(uuid)
        file = asset.content.file
        meta = _describe(asset)

        # Refuse the wrong kind of file (and an empty/oversized one) *before*
        # downloading or calling the provider — the same discipline `view` follows.
        if not _is_audio(file.content_type):
            return f"{meta}\n(not an audio file — 'listen' is for audio. Use 'read' for text, 'view' for images.)"
        if file.byte_size <= 0:
            return f"{meta}\n(empty file — nothing to hear.)"
        if file.byte_size > MAX_AUDIO_BYTES:
            return (
                f"{meta}\n({file.byte_size} bytes, over the {MAX_AUDIO_BYTES}-byte "
                "transcription limit — too large to listen to.)"
            )

        key = self._api_key or os.environ.get("AI_API_KEY")
        if not key:
            return (
                "Error: no API key for transcription. Set AI_API_KEY "
                "(or pass api_key= to HearAudioTool)."
            )

        data = _download(file.url)
        try:
            transcript = self._transcribe(data, file.filename, file.content_type, key)
        except ProviderConnectionError as exc:
            return f"Error transcribing audio: could not reach the transcription API: {exc}"
        except ProviderError as exc:
            return f"Error transcribing audio: {exc}"

        if not transcript.strip():
            return f"{meta}\n(transcribed, but no speech was detected.)"
        return f"{meta}\n\nTranscript:\n{transcript}"

    def _transcribe(self, data: bytes, filename: str, content_type: str, key: str) -> str:
        """Send the audio to the transcription endpoint (via the openai SDK) and return its text.

        Failures surface as the same typed `ProviderError`s the model adapter raises (mapped
        from the SDK's exceptions by `sdk_error_context`) — `run` catches them and relays the
        message to the model. The audio rides as an in-memory ``(filename, bytes, content_type)``
        file the SDK uploads, so the bytes never touch the filesystem.
        """
        openai = require_openai_sdk()
        client = openai.OpenAI(api_key=key, base_url=self._base_url, timeout=self._timeout)
        with sdk_error_context(openai):
            response = client.audio.transcriptions.create(
                model=self._model, file=(filename, data, content_type)
            )
        return _extract_transcript(response)


def _extract_transcript(response: object) -> str:
    """Pull the transcript text out of an SDK transcription result.

    The default (``json``) format gives a ``Transcription`` whose ``.text`` is the transcript.
    Anything else — a bare string the SDK handed back from a non-standard (e.g. proxy) body, or
    a result with no string ``text`` — is treated as a provider error rather than guessed at, so
    even a malformed response surfaces as model-readable text, never a traceback.
    """
    text = getattr(response, "text", None)
    if not isinstance(text, str):
        raise ProviderError("the transcription API returned no text.")
    return text
