"""Let the agent hear: transcribe an audio asset so it can read what was said.

The audio third of the three senses the assets tool carries — `view` an image, `watch` a video,
`listen` to audio (issue #484). On a platform that carries audio — TTS, music, voice notes — a peer
that cannot hear is half-deaf; `listen` closes that gap by turning an audio asset into text the
model can read and reason over.

Why the *transcription* lives here and not in `_assets.py`
----------------------------------------------------------
The other two senses are pure platform I/O: `view` and `watch` download a file and hand the bytes
to the engine, which puts them in front of the model. Hearing is different — it needs a **provider
call** (OpenAI's transcription endpoint), exactly like image *generation*. So the call follows
`GenerateImageTool`'s shape: it holds the agent's ``AI_API_KEY`` and reaches OpenAI's Audio
endpoint **through the ``openai`` SDK** (``client.audio.transcriptions``), never hand-rolled HTTP —
the same vendor-SDK rule the model loop follows (issue #158), keeping the brain/body boundary clean
(the platform SDK never reaches the model provider, and vice versa).

That difference is also why `listen` is the one assets action that can be **absent**: an agent with
no transcription provider configured never sees it in the schema at all (`_assets.assets_options`,
the "don't show a locked door" rule). The other two senses cost no provider call, so they are
always there.

`Transcriber` is what the assets tool holds when it can hear. It mirrors `view`'s on-demand,
ephemeral shape: the agent listens only when it chooses (never eagerly inlined), a non-audio asset
comes back as a clean note rather than a failure, and an oversized one is described, not force-fed.
The transcription model is OpenAI's Audio API, sharing the agent's one key (``gpt-5.4-mini``
reasons, ``gpt-image-2.5`` paints, ``gpt-transcribe`` listens).
"""

from __future__ import annotations

import os

from basecradle_harness._exceptions import ProviderError
from basecradle_harness._observability import media_timer
from basecradle_harness._openai import require_openai_sdk, sdk_error_context

#: OpenAI's Audio API root. Transcription is an OpenAI service; this changes only for
#: a proxy, not to reach another vendor (the key is the OpenAI key).
DEFAULT_BASE_URL = "https://api.openai.com/v1"
#: The transcription model. ``gpt-transcribe`` is OpenAI's current speech-to-text model;
#: ``gpt-4o-transcribe`` and ``whisper-1`` are the older alternatives the same endpoint
#: accepts. Not ``gpt-live-transcribe`` — that is the *streaming* variant, and this
#: transcribes a whole file in one call.
DEFAULT_MODEL = "gpt-transcribe"
#: Transcription is slow next to a chat call — give it room before giving up.
DEFAULT_TIMEOUT = 120.0
#: The largest audio file to transcribe. 25 MiB is OpenAI's per-file upload ceiling;
#: larger than this is described, not sent (it would only be rejected downstream).
MAX_AUDIO_BYTES = 25 * 1024 * 1024


class Transcriber:
    """The transcription provider call behind the assets tool's ``listen`` action.

    A small collaborator rather than a tool: the assets tool owns the platform I/O (fetch the
    asset, refuse the wrong kind of file, render the result), and this owns the one thing that is
    not platform I/O — the provider call. Splitting them that way is what let `listen` become an
    *action* on `assets` without the assets tool growing a vendor SDK import at module scope.

    Args:
        api_key: The OpenAI key for the Audio API. Falls back to ``AI_API_KEY`` at call
            time, so constructing it needs no secret (a keyless construction just reports,
            readably, when used).
        base_url: The Audio API root. Defaults to OpenAI.
        model: The transcription model. Defaults to ``gpt-transcribe``.
        timeout: Per-request timeout in seconds (transcription is slow).
    """

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

    @property
    def key(self) -> str | None:
        """The key this transcriber would use, read at call time — ``None`` when there is none."""
        return self._api_key or os.environ.get("AI_API_KEY")

    def transcribe(self, data: bytes, filename: str, content_type: str, key: str) -> str:
        """Send the audio to the transcription endpoint (via the openai SDK) and return its text.

        Failures surface as the same typed `ProviderError`s the model adapter raises (mapped
        from the SDK's exceptions by `sdk_error_context`) — the caller catches them and relays the
        message to the model. The audio rides as an in-memory ``(filename, bytes, content_type)``
        file the SDK uploads, so the bytes never touch the filesystem.
        """
        openai = require_openai_sdk()
        client = openai.OpenAI(api_key=key, base_url=self._base_url, timeout=self._timeout)
        with (
            media_timer(provider="openai", kind="audio.transcribe", model=self._model),
            sdk_error_context(openai),
        ):
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
