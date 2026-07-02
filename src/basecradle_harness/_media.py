"""Vendor-neutral plumbing the image and video tools share: error relay, sniffing, filenames.

These three helpers have nothing to do with *which* vendor renders the pixels, so they live
here rather than in either `_images.py` (OpenAI) or `_grok.py` (xAI) — letting the xAI media
tools stay cleanly separate from the OpenAI ones while still sharing the parts that are
genuinely common:

- `provider_error_message` — dig the *real* cause out of a provider HTTP error so the model
  relays it, never an opaque "HTTP 400" (the tool-building discipline's Principle 5);
- `sniff_media_ext` — read a blob's true format from its magic bytes, so the uploaded Asset's
  filename extension (and thus its server-inferred content-type) matches the actual bytes —
  the bug-proof generalization of the old hard-coded ``.png``;
- `media_filename` / `slugify` — a safe, readable filename whose extension is the real one.
"""

from __future__ import annotations

import base64
import json
import re
from collections.abc import Callable

from basecradle_harness._exceptions import ProviderAPIError, ProviderError


def provider_error_message(exc: ProviderError, subject: str) -> str:
    """A legible relay of a media-API failure, naming `subject` (e.g. ``"the xAI image API"``).

    A `ProviderAPIError`'s own ``str`` is only the generic "Provider returned HTTP 400" — the
    *real* cause lives in its response ``body``, where the API puts it under ``error.message``
    (an object) or ``error`` (a bare string), or occasionally a top-level ``message``/``msg``.
    Surface that so the AI passes the true cause to the user instead of an opaque status. A
    plain `ProviderError` (e.g. a 200 carrying no image) already has a useful ``str``.
    """
    if not isinstance(exc, ProviderAPIError):
        return str(exc)
    body = exc.body or ""
    detail = _detail_from_body(body) if body else ""
    return f"{subject} rejected the request — {detail or exc}"


def _detail_from_body(body: str) -> str:
    """The human-readable error detail in a provider error body, across the common shapes."""
    try:
        data = json.loads(body)
    except ValueError:
        return body.strip()
    if not isinstance(data, dict):
        return body.strip()
    error = data.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or error.get("msg") or "").strip() or body.strip()
    if isinstance(error, str) and error.strip():
        return error.strip()
    for key in ("message", "msg", "detail"):
        value = data.get(key)
        if value:
            return str(value).strip()
    return body.strip()


def decode_image_payload(
    body: dict, *, download: Callable[[str], bytes], subject: str = "the image API"
) -> bytes:
    """The image bytes from an OpenAI-shaped Images response: ``b64_json`` or a ``url``.

    The endpoint returns the first ``data`` item either inline (``b64_json``, the default we
    request) or as a ``url`` (when an agent asked for that encoding) — handle both so the tool
    never silently drops a valid result. `download` is injected so this module stays free of an
    HTTP client; the image tool passes the shared blob fetcher.
    """
    items = body.get("data") or []
    item = items[0] if isinstance(items, list) and items else None
    if isinstance(item, dict):
        encoded = item.get("b64_json")
        if encoded:
            try:
                return base64.b64decode(encoded)
            except (ValueError, TypeError) as exc:
                raise ProviderError(f"{subject} returned undecodable data: {exc}") from exc
        url = item.get("url")
        if url:
            return download(url)
    raise ProviderError(f"{subject} returned no image data.")


def sniff_media_ext(data: bytes, default: str) -> str:
    """The file extension for `data`, read from its magic bytes; `default` if unrecognized.

    The uploaded Asset's content-type is inferred from its filename, so the extension must
    follow the *actual* bytes — sniffing makes that impossible to get wrong regardless of what
    a vendor's ``output_format``/``response_format`` knob claims (the hard-coded-``.png`` bug,
    generalized away). Covers the image and video formats these tools emit.
    """
    if data[:4] == b"\x89PNG":
        return "png"
    if data[:3] == b"\xff\xd8\xff":
        return "jpg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if data[4:8] == b"ftyp":  # ISO base media (mp4 / m4v / mov family)
        return "mp4"
    if data[:4] == b"\x1aE\xdf\xa3":  # EBML — Matroska / WebM
        return "webm"
    return default


def uuid_list(value: list[str] | str | None) -> list[str]:
    """Normalize an ``image`` arg to a clean list of asset uuids, for the edit tools.

    The schema declares an array of source uuids, but a model may pass a bare string for a
    single source; accept both, and drop any blank entries so an empty/whitespace value
    surfaces the friendly "needs a source" error rather than a 400 deep in the API. Shared by
    the OpenAI (`_images.EditImageTool`) and xAI (`_grok.GrokEditImageTool`) edit tools so the
    two normalize an identical arg identically.
    """
    if value is None:
        return []
    items = [value] if isinstance(value, str) else list(value)
    return [u.strip() for u in items if isinstance(u, str) and u.strip()]


def slugify(text: str) -> str:
    """A short, filename-safe slug: lowercased words joined by hyphens, capped at 48 chars."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return "-".join(words)[:48].strip("-")


def media_filename(filename: str | None, fallback_text: str, ext: str) -> str:
    """A safe filename ending in `ext`: the operator's name if given, else a slug of the prompt.

    Assets are addressed by their own uuid, so two files sharing a name never collide — a
    readable name is for humans, not uniqueness. The extension is the sniffed, real one.
    """
    base = filename.rsplit(".", 1)[0] if filename else fallback_text
    return f"{slugify(base) or 'media'}.{ext}"
