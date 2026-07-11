"""Make media the xAI way: generate an image, or a video, with grok and post it as an asset.

The xAI-native counterpart to the OpenAI image tools (`_images.py`). These are **Eddie
Murphy's** media hands — a fully-xAI persona — so they live in their own module, talk only to
``api.x.ai``, and carry the agent's ``AI_API_KEY`` (an xAI key for an xAI agent). Two
tools, split by operation, the tool-building discipline (full surface → coverage decided →
split by operation → every option tested):

- `GrokGenerateImageTool` (``grok_generate_image``) — text → image, via xAI's OpenAI-shaped
  Images endpoint (``POST /v1/images/generations``, ``grok-imagine-image-quality``).
- `GrokEditImageTool` (``grok_edit_image``) — image(s) → image, via xAI's image-edit endpoint
  (``POST /v1/images/edits``, ``grok-imagine-image-quality``). The xAI-native counterpart to
  the OpenAI `edit_image` (``_images.py``). Two asymmetries vs OpenAI, both deliberate and
  documented: (1) the OpenAI SDK's ``images.edit()`` is **not** usable — it sends
  ``multipart/form-data`` and xAI requires ``application/json`` (xAI's docs say so explicitly),
  so this tool posts JSON over the shared grok transport, sending each source image as a
  **base64 data URI** (the signed Asset URL is not assumed publicly fetchable by xAI); (2) xAI
  does **natural-language** editing with **no mask** (no mask-based inpainting), so — unlike
  OpenAI `edit_image` — there is no ``mask`` parameter. xAI composites up to **3** source
  images. The request shape is ``image`` (a single ``{"type":"image_url","url":…}`` object) for
  one source, or ``images`` (an array of them) for a composite (docs.x.ai images/editing +
  multi-image-editing).
- `GrokGenerateVideoTool` (``grok_generate_video``) — text → video **or** image → video, via
  xAI's **asynchronous** video endpoint (``POST /v1/videos/generations``, ``grok-imagine-video``).
  This is the harness's first video capability. Generation takes minutes: the call returns a
  ``request_id`` and the tool polls ``GET /v1/videos/{request_id}`` until the clip is ``done``,
  then downloads the produced ``.mp4`` and uploads it as an Asset that renders inline in the UI.

Why function tools, not a provider built-in
-------------------------------------------
Same boundary the OpenAI image tools keep: the provider is the *brain* and has no business
reaching the *body*'s SDK. A `PlatformTool` holds the live SDK client, owns the xAI HTTP, and
uploads the result — so the provider stays a pure text/tool-call adapter and the capability
composes under whichever provider Eddie runs. A new capability is one small tool class.

Coverage (audited to grok-imagine's surface)
--------------------------------------------
Image: ``aspect_ratio`` and ``resolution`` are exposed as optional pass-throughs; the default
call sends only ``model`` + ``prompt`` (+ ``response_format=b64_json`` so we get bytes), which
is the always-valid core. ``n>1`` is deliberately skipped — multiple-images-per-call is niche
for a conversational agent (founder decision, consistent with the OpenAI image tool). Edit:
just ``model`` + ``prompt`` + the source ``image``/``images`` (+ ``response_format=b64_json``);
no ``aspect_ratio``/``resolution`` (an edit follows the source's dimensions) and no ``mask``
(xAI edits by natural language, not a mask region) — the documented asymmetries vs OpenAI. Video:
``duration``, ``aspect_ratio``, ``resolution`` are exposed, plus ``image`` (a source Asset
uuid) for image-to-video. xAI enforces the enum/range constraints, so they are documented in
the schema rather than re-validated here — coverage never drifts as the model's surface
evolves, and an out-of-range value comes back as a *legible* xAI error, not a generic 400.
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx
from basecradle import BaseCradleError

from basecradle_harness._assets import _data_url, _describe, _download, _upload
from basecradle_harness._exceptions import ProviderConnectionError, ProviderError
from basecradle_harness._http import raise_for_status
from basecradle_harness._media import (
    decode_image_payload,
    media_filename,
    provider_error_message,
    sniff_media_ext,
    uuid_list,
)
from basecradle_harness._observability import media_timer
from basecradle_harness._platform import PlatformTool, explain

#: xAI's API root. These tools are xAI-native; this changes only for a proxy, never to reach
#: another vendor (the key is the agent's xAI key).
DEFAULT_BASE_URL = "https://api.x.ai/v1"
#: The grok image model — xAI's quality image tier.
DEFAULT_IMAGE_MODEL = "grok-imagine-image-quality"
#: The grok video model. ``grok-imagine-video-1.5`` is the newer alternative an operator can pass.
DEFAULT_VIDEO_MODEL = "grok-imagine-video"
#: Per-request HTTP timeout. Generous — a generation submit/poll call is slower than a chat
#: call, and ``grok_edit_image`` runs the same class of slow, high-fidelity image-edit work
#: (``grok-imagine-image-quality``) that a measured ~133s ``gpt-image-2`` ``quality: high``
#: edit timed out under a 120s ceiling (issue #222, sibling of #219). 300s clears that class
#: of worst case with headroom; a timeout is a ceiling, not a fixed wait, so it costs nothing
#: on fast calls.
DEFAULT_TIMEOUT = 300.0
#: Video polling: how often to re-check, and the ceiling before giving up. Video generation
#: "typically takes up to several minutes" (xAI), so the default ceiling is roomy.
DEFAULT_POLL_INTERVAL = 5.0
DEFAULT_POLL_MAX_WAIT = 600.0


class _GrokMediaTool(PlatformTool):
    """Shared base for the grok media tools: the xAI key, the HTTP plumbing, the upload.

    Each subclass is a `PlatformTool` (so it holds the live SDK client and uploads through it)
    that owns one xAI media endpoint. The base carries the key resolution and the JSON
    request/poll helpers; the subclasses own their parameter schema and their endpoint flow.
    """

    #: The default model for this tool, used when the constructor is given none.
    default_model: str = ""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        model: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model or self.default_model
        self._timeout = timeout

    def _key(self) -> str | None:
        """The xAI key: the explicit one, or the agent's provider key from the environment."""
        return self._api_key or os.environ.get("AI_API_KEY")

    def _request(
        self, key: str, method: str, endpoint: str, *, json: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Call an xAI endpoint and return the decoded JSON response.

        One helper for the submit (POST) and the video poll (GET): a fresh client per call
        (these are infrequent generation calls, not a hot path), connection failures mapped to
        `ProviderConnectionError`, and any 4xx/5xx to a typed `ProviderAPIError` carrying the
        body so the real cause is relayed.
        """
        try:
            with httpx.Client(
                headers={"Authorization": f"Bearer {key}"}, timeout=self._timeout
            ) as client:
                response = client.request(method, f"{self._base_url}/{endpoint}", json=json)
        except httpx.RequestError as exc:
            raise ProviderConnectionError(str(exc)) from exc
        if response.status_code >= 400:
            raise_for_status(response)
        return response.json()

    def _post_asset(
        self, timeline: str, data: bytes, name: str, description: str, verb: str
    ) -> str:
        """Upload the produced bytes as an Asset and report it for the model to read."""
        asset = _upload(self.context.client, timeline, data, name, description)
        return f"{verb} {name!r} ({len(data)} bytes). {_describe(asset)}"

    def _source_file(self, uuid: str) -> Any:
        """Resolve a source Asset uuid to its file object, relaying a bad uuid legibly.

        Shared by the tools that take a source image (edit → data URI, image-to-video → blob
        URL): both need the same `assets.get` and the same ``BaseCradleError → ProviderError``
        relay, so the AI learns *why* a source couldn't be read rather than seeing a crash.
        """
        try:
            return self.context.client.assets.get(uuid).content.file
        except BaseCradleError as error:
            raise ProviderError(
                f"couldn't read source image asset {uuid!r}: {explain(error)}"
            ) from error


class GrokGenerateImageTool(_GrokMediaTool):
    """``grok_generate_image`` — render an image from a prompt with grok and post it."""

    name = "grok_generate_image"
    description = (
        "Create an image from a text prompt with xAI's grok image model and post it as a file "
        "on the timeline, where it renders inline. Use for 'draw…', 'make a picture of…', "
        "'generate an image of…'. Returns the new asset's uuid."
    )
    parameters = {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "What to draw — a clear, specific description of the image.",
            },
            "aspect_ratio": {
                "type": "string",
                "description": (
                    "Optional aspect ratio, e.g. '16:9', '1:1', '9:16', '4:3', '3:4', '3:2', "
                    "'2:3'. Omit to let the model choose its default."
                ),
            },
            "resolution": {
                "type": "string",
                "description": "Optional resolution hint, e.g. '480p' or '720p'. Omit for the default.",
            },
            "filename": {
                "type": "string",
                "description": "Optional filename for the posted image (extension follows the real format).",
            },
            "description": {
                "type": "string",
                "description": "Optional human-readable description stored with the asset.",
            },
            "timeline": {
                "type": "string",
                "description": "Optional timeline uuid to post to. Defaults to the current timeline.",
            },
        },
        "required": ["prompt"],
    }

    default_model = DEFAULT_IMAGE_MODEL

    def run(
        self,
        prompt: str,
        aspect_ratio: str | None = None,
        resolution: str | None = None,
        filename: str | None = None,
        description: str | None = None,
        timeline: str | None = None,
    ) -> str:
        """Generate the image, upload it, and report the new asset for the model to read."""
        if not prompt or not prompt.strip():
            return "Error: 'grok_generate_image' needs a 'prompt' describing what to draw."
        key = self._key()
        if not key:
            return "Error: no API key for image generation. Set AI_API_KEY to the agent's xAI key."

        # response_format=b64_json so the image comes back inline (no second fetch); aspect_ratio
        # and resolution are sent only when set, so the default call is always the valid core.
        payload: dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "n": 1,
            "response_format": "b64_json",
        }
        if aspect_ratio:
            payload["aspect_ratio"] = aspect_ratio
        if resolution:
            payload["resolution"] = resolution

        try:
            with media_timer(provider="xai", kind="image.generate", model=self._model):
                body = self._request(key, "POST", "images/generations", json=payload)
            image_bytes = decode_image_payload(
                body, download=_download, subject="the xAI image API"
            )
        except ProviderConnectionError as exc:
            return f"Error generating image: could not reach the xAI image API: {exc}"
        except ProviderError as exc:
            return f"Error generating image: {provider_error_message(exc, 'the xAI image API')}"

        target = timeline or self.context.timeline
        ext = sniff_media_ext(image_bytes, "jpg")
        name = media_filename(filename, prompt, ext)
        return self._post_asset(
            target,
            image_bytes,
            name,
            description or f"Generated image: {prompt}",
            "Generated and posted",
        )


class GrokEditImageTool(_GrokMediaTool):
    """``grok_edit_image`` — edit one or more existing image assets with a prompt and post it.

    The xAI-native counterpart to the OpenAI `EditImageTool`. Where OpenAI's edit endpoint takes
    raw multipart bytes, xAI's takes **JSON** with each source as a base64 **data URI** — so the
    tool resolves each source Asset by uuid, downloads its bytes through the bound SDK client's
    signed blob URL, and inlines them as ``data:<type>;base64,…``. A single source rides the
    ``image`` object; two or three composite via the ``images`` array. There is **no mask** —
    xAI edits by natural language, not a mask-based region (documented asymmetry vs OpenAI).
    """

    name = "grok_edit_image"
    description = (
        "Edit an image that already exists on the timeline with xAI's grok image model and post "
        "the result as a new file, the way a peer marks up or restyles a picture. Give one or "
        "more source asset uuids in 'image' (find them with the assets tool's 'list') and "
        "describe the change in 'prompt' (e.g. 'recolor the car red', or composite several "
        "sources — up to 3). Unlike editing with a mask, grok edits by natural language, so "
        "there is no mask region. To make a brand-new image from text instead, use "
        "'grok_generate_image'. Returns the new asset's uuid."
    )
    parameters = {
        "type": "object",
        "properties": {
            "image": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "One or more source image asset uuids to edit. Get them from the assets "
                    "tool's 'list'. Pass several (up to 3) to composite them into one image."
                ),
            },
            "prompt": {
                "type": "string",
                "description": "How to change the image — describe the edit you want.",
            },
            "filename": {
                "type": "string",
                "description": "Optional filename for the posted image (extension follows the real format).",
            },
            "description": {
                "type": "string",
                "description": "Optional human-readable description stored with the asset.",
            },
            "timeline": {
                "type": "string",
                "description": "Optional timeline uuid to post to. Defaults to the current timeline.",
            },
        },
        "required": ["image", "prompt"],
    }

    default_model = DEFAULT_IMAGE_MODEL

    def run(
        self,
        image: list[str] | str | None = None,
        prompt: str | None = None,
        filename: str | None = None,
        description: str | None = None,
        timeline: str | None = None,
    ) -> str:
        """Edit the source image(s), upload the result, and report the new asset for the model."""
        if not prompt or not prompt.strip():
            return "Error: 'grok_edit_image' needs a 'prompt' describing the change to make."
        uuids = uuid_list(image)
        if not uuids:
            return (
                "Error: 'grok_edit_image' needs at least one source 'image' asset uuid. "
                "Use the assets tool's 'list' to find one."
            )
        key = self._key()
        if not key:
            return "Error: no API key for image editing. Set AI_API_KEY to the agent's xAI key."

        # response_format=b64_json so the edited image comes back inline (no second fetch);
        # decode_image_payload handles a url response too, so a vendor default never drops it.
        payload: dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "n": 1,
            "response_format": "b64_json",
        }
        try:
            sources = [self._source_image(uuid) for uuid in uuids]
            # One source rides the ``image`` object; a composite (2-3) rides the ``images`` array
            # — the two request shapes xAI's single- and multi-image edit endpoints document.
            if len(sources) == 1:
                payload["image"] = sources[0]
            else:
                payload["images"] = sources
            with media_timer(provider="xai", kind="image.edit", model=self._model):
                body = self._request(key, "POST", "images/edits", json=payload)
            image_bytes = decode_image_payload(
                body, download=_download, subject="the xAI image API"
            )
        except ProviderConnectionError as exc:
            return f"Error editing image: could not reach the xAI image API: {exc}"
        except ProviderError as exc:
            return f"Error editing image: {provider_error_message(exc, 'the xAI image API')}"

        target = timeline or self.context.timeline
        ext = sniff_media_ext(image_bytes, "jpg")
        name = media_filename(filename, prompt, ext)
        return self._post_asset(
            target,
            image_bytes,
            name,
            description or f"Edited image: {prompt}",
            "Edited and posted",
        )

    def _source_image(self, uuid: str) -> dict[str, str]:
        """Resolve a source image Asset uuid to an xAI ``image_url`` object (a base64 data URI).

        xAI's edit endpoint takes the image inline as a data URI rather than a fetchable URL — the
        signed Asset URL is not assumed publicly reachable by xAI's servers — so the bytes are
        downloaded and base64-encoded. A bad uuid (or a download failure) raises a legible
        `ProviderError` the caller relays, so the AI learns *why* the source couldn't be read.
        """
        file = self._source_file(uuid)
        try:
            data = _download(file.url)
        except httpx.HTTPError as exc:
            raise ProviderError(f"couldn't download source image asset {uuid!r}: {exc}") from exc
        # A blob missing its content-type would crash `_data_url` (it splits the string) — fall
        # back to a generic type so a malformed asset still relays legibly, never AttributeErrors.
        return {
            "type": "image_url",
            "url": _data_url(file.content_type or "application/octet-stream", data),
        }


class GrokGenerateVideoTool(_GrokMediaTool):
    """``grok_generate_video`` — generate a video (text→video or image→video) and post it.

    xAI's video endpoint is asynchronous: the submit returns a ``request_id`` and the tool
    polls until the clip is ``done``, then downloads and uploads it. ``poll_interval`` and
    ``poll_max_wait`` are constructor knobs so a test can drive the poll loop without sleeping.
    """

    name = "grok_generate_video"
    description = (
        "Create a short video with xAI's grok video model and post it as a file on the "
        "timeline, where it renders inline. Text-to-video from a 'prompt', or image-to-video "
        "by also passing 'image' (a source image asset's uuid) to animate it. Generation takes "
        "a minute or two. Returns the new asset's uuid."
    )
    parameters = {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "What should happen in the video — motion, scene, camera.",
            },
            "image": {
                "type": "string",
                "description": (
                    "Optional source image asset uuid to animate (image-to-video). The video "
                    "starts from this frame; omit for text-to-video."
                ),
            },
            "duration": {
                "type": "integer",
                "description": "Optional clip length in seconds (e.g. 1–15). Omit for the model default.",
            },
            "aspect_ratio": {
                "type": "string",
                "description": (
                    "Optional aspect ratio, e.g. '16:9', '1:1', '9:16', '4:3', '3:4', '3:2', "
                    "'2:3'. Omit for the default."
                ),
            },
            "resolution": {
                "type": "string",
                "description": "Optional resolution, e.g. '480p' or '720p'. Omit for the default.",
            },
            "filename": {
                "type": "string",
                "description": "Optional filename for the posted video (extension follows the real format).",
            },
            "description": {
                "type": "string",
                "description": "Optional human-readable description stored with the asset.",
            },
            "timeline": {
                "type": "string",
                "description": "Optional timeline uuid to post to. Defaults to the current timeline.",
            },
        },
        "required": ["prompt"],
    }

    default_model = DEFAULT_VIDEO_MODEL

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        model: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        poll_max_wait: float = DEFAULT_POLL_MAX_WAIT,
    ) -> None:
        super().__init__(api_key=api_key, base_url=base_url, model=model, timeout=timeout)
        self._poll_interval = poll_interval
        self._poll_max_wait = poll_max_wait

    def run(
        self,
        prompt: str,
        image: str | None = None,
        duration: int | None = None,
        aspect_ratio: str | None = None,
        resolution: str | None = None,
        filename: str | None = None,
        description: str | None = None,
        timeline: str | None = None,
    ) -> str:
        """Submit the video job, poll it to completion, upload the clip, and report the asset."""
        if not prompt or not prompt.strip():
            return "Error: 'grok_generate_video' needs a 'prompt' describing the video."
        key = self._key()
        if not key:
            return "Error: no API key for video generation. Set AI_API_KEY to the agent's xAI key."

        payload: dict[str, Any] = {"model": self._model, "prompt": prompt}
        if duration is not None:
            payload["duration"] = duration
        if aspect_ratio:
            payload["aspect_ratio"] = aspect_ratio
        if resolution:
            payload["resolution"] = resolution

        try:
            if image:
                payload["image_url"] = self._source_image_url(image)
            # The timed span is submit → done (the poll loop *is* the generation on this
            # endpoint); the clip download that follows is transfer, not model time.
            with media_timer(provider="xai", kind="video.generate", model=self._model):
                request_id = self._submit(key, payload)
                video_url = self._await_video(key, request_id)
            video_bytes = _download(video_url)
        except ProviderConnectionError as exc:
            return f"Error generating video: could not reach the xAI video API: {exc}"
        except ProviderError as exc:
            return f"Error generating video: {provider_error_message(exc, 'the xAI video API')}"
        except httpx.HTTPError as exc:
            return f"Error generating video: couldn't download the produced video: {exc}"

        target = timeline or self.context.timeline
        ext = sniff_media_ext(video_bytes, "mp4")
        name = media_filename(filename, prompt, ext)
        return self._post_asset(
            target,
            video_bytes,
            name,
            description or f"Generated video: {prompt}",
            "Generated and posted",
        )

    def _source_image_url(self, uuid: str) -> str:
        """Resolve a source image Asset uuid to its dereferenceable blob URL.

        xAI's image-to-video takes an ``image_url`` (not bytes), and the platform blob URL is
        already authorized, so xAI's servers can fetch it. A bad uuid raises a legible
        `ProviderError` the caller relays — the AI learns *why* the source couldn't be read.
        """
        return self._source_file(uuid).url

    def _submit(self, key: str, payload: dict[str, Any]) -> str:
        """Submit the generation job and return its ``request_id``."""
        body = self._request(key, "POST", "videos/generations", json=payload)
        request_id = body.get("request_id") or body.get("id")
        if not request_id:
            raise ProviderError("the xAI video API did not return a request_id.")
        return str(request_id)

    def _await_video(self, key: str, request_id: str) -> str:
        """Poll the job until it is ``done`` and return the produced video URL.

        Raises a legible `ProviderError` if the job ``failed``/``expired`` (relaying xAI's own
        message) or does not finish within ``poll_max_wait`` — never a silent hang or an opaque
        status, so the model can tell the user what happened.
        """
        deadline = time.monotonic() + self._poll_max_wait
        while True:
            body = self._request(key, "GET", f"videos/{request_id}")
            status = body.get("status")
            if status == "done":
                video = body.get("video")
                url = video.get("url") if isinstance(video, dict) else None
                if not url:
                    raise ProviderError("the xAI video API finished but returned no video url.")
                return str(url)
            if status in ("failed", "expired"):
                error = body.get("error")
                detail = error.get("message") if isinstance(error, dict) else error
                raise ProviderError(f"the xAI video API {status}: {detail or 'no detail given'}")
            if time.monotonic() >= deadline:
                raise ProviderError(
                    f"the xAI video API did not finish within {int(self._poll_max_wait)}s "
                    f"(last status {status!r})."
                )
            time.sleep(self._poll_interval)
