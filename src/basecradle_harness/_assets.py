"""Give the agent files: list, read, view, and create assets on a BaseCradle timeline.

This is the first tool that acts *on* the platform, so it is the first
`PlatformTool` — it reaches the SDK client and the current timeline through the
bound `PlatformContext` (see `_platform.py`). Everything else is the same small
contract `MemoryTool` follows: an `action` enum, branching in `run`, a string
back for the model to read. A contributor adds the next platform tranche (tasks,
participants, …) by copying this shape.

Four actions, the file equivalent of what a human peer does on a timeline:

- **list** — what files are here, with the uuids needed to read them.
- **read** — download one file and surface it. The model is text, so a text-ish
  file comes back decoded; a binary (or oversized) file comes back as metadata
  plus a "not inlined" note rather than a wall of bytes dumped into context.
- **view** — *look at* an image file. Where `read` refuses a binary, `view`
  fetches an image and hands it back as a `ToolResult` carrying the picture, so a
  vision-capable model (the Responses path) actually sees it. This is the
  on-demand "look at this asset" step — images are never inlined eagerly, only
  when the agent chooses to look.
- **create** — upload content the agent produced (the common path: text → file),
  with an optional description.

Ops default to the **current** timeline (the one the agent is engaged on); an
explicit `timeline` uuid handles the rare cross-timeline case.

I/O discipline (safe-by-construction): the SDK is the only platform I/O, and
nothing touches the filesystem. A read decodes the downloaded bytes in memory; a
create streams the produced text straight to the SDK from an in-memory buffer.
There are no temp files to confine or clean up — the strongest version of "keep
scratch under `HARNESS_HOME` and clean up" is to never write scratch at all.
"""

from __future__ import annotations

import base64
import io
import itertools
from typing import Any

import httpx

from basecradle_harness._messages import ImageContent, ToolResult
from basecradle_harness._platform import PlatformTool

# A text-ish file is decoded and inlined; anything else is described, not dumped.
# `byte_size` over this cap is treated as not-inlinable even when text, so a huge
# log never blows up the model's context. 256 KiB is generous for prose/code.
MAX_INLINE_BYTES = 256 * 1024

# The largest image `view` will inline as model input. Vision images are heavier
# than text but the model can handle a real photo; 20 MiB matches OpenAI's
# per-image input ceiling. Larger than this is described, not shown.
MAX_IMAGE_BYTES = 20 * 1024 * 1024

# Image content types a vision model can take as input. Kept to the formats the
# model providers document so `view` gives a clean "can't show that" rather than
# letting an unsupported type fail deep in the provider call.
_VIEWABLE_IMAGE_TYPES = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})

# How many assets one `list` returns. A timeline rarely has more; the cap keeps a
# pathological one from flooding context. When it bites, the reply says so.
DEFAULT_LIST_LIMIT = 50

# Download timeout (seconds) for a blob fetch (`read`, `view`, and audio `listen`).
# The URL is a dereferenceable blob link; 30s also bounds a larger audio fetch (up
# to the 25 MiB transcription ceiling), which on a slow link returns a clean error.
_DOWNLOAD_TIMEOUT = 30.0

# `application/*` content types that are really text. `text/*`, `*+json`, and
# `*+xml` are matched structurally in `_is_text`; these are the exceptions.
_TEXTUAL_APPLICATION_TYPES = frozenset(
    {
        "application/json",
        "application/xml",
        "application/javascript",
        "application/x-ndjson",
        "application/x-yaml",
        "application/yaml",
        "application/toml",
        "application/csv",
        "application/sql",
        "application/x-sh",
    }
)


class AssetsTool(PlatformTool):
    """List, read, view, and create files (assets) on the agent's current timeline.

    A `PlatformTool`: the hosting agent binds the SDK client and current-timeline
    uuid before the loop runs. Until bound, `run` reports it is not connected
    (via `PlatformError`) rather than failing obscurely.
    """

    name = "assets"
    description = (
        "Exchange files on the timeline, the way a peer shares an attachment. "
        "action='list' shows the files here with their uuids; action='read' "
        "downloads one file by uuid and returns its text (binary files come back as "
        "a description, not raw bytes); action='view' looks at an image file by uuid "
        "(or pass uuid='latest' to view the most recent file on the timeline, such as "
        "an image you just posted) so you can actually see it and describe or reason "
        "about it; action='create' "
        "uploads a new file from the text content you provide, with a filename and an "
        "optional description. Operations use the current timeline unless you pass a "
        "timeline uuid. "
        "Assets are shared with every viewer and can never be edited or deleted — prefer "
        "your own storage for private or working files; upload what is meant for the peers here."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "read", "view", "create"],
                "description": "What to do.",
            },
            "uuid": {
                "type": "string",
                "description": (
                    "The asset's uuid (read/view only). Get it from 'list', or pass "
                    "'latest' for the most recent file on the timeline — e.g. an image "
                    "you just generated and posted, so you can view it without being "
                    "handed its uuid."
                ),
            },
            "content": {
                "type": "string",
                "description": "The text content of the file to upload (create only).",
            },
            "filename": {
                "type": "string",
                "description": "The filename for the uploaded file, e.g. 'notes.md' (create only).",
            },
            "description": {
                "type": "string",
                "description": "An optional human-readable description of the file (create only).",
            },
            "timeline": {
                "type": "string",
                "description": (
                    "Optional timeline uuid to act on instead of the current one. "
                    "Omit to use the timeline you are engaged on."
                ),
            },
        },
        "required": ["action"],
    }

    def run(
        self,
        action: str,
        uuid: str | None = None,
        content: str | None = None,
        filename: str | None = None,
        description: str | None = None,
        timeline: str | None = None,
    ) -> str | ToolResult:
        """Dispatch on `action`. Returns a message written for the model to read.

        `view` may return a `ToolResult` carrying the image for the model to see;
        every other action returns a plain string.
        """
        target = timeline or self.context.timeline
        if action == "list":
            return self._list(target)
        if action == "read":
            if not uuid:
                return "Error: 'read' needs the asset's uuid. Use 'list' to find it."
            resolved = self._resolve_uuid(uuid, target)
            if resolved is None:
                return "No files on this timeline yet — nothing to read."
            return self._read(resolved)
        if action == "view":
            if not uuid:
                return "Error: 'view' needs the asset's uuid. Use 'list' to find it."
            resolved = self._resolve_uuid(uuid, target)
            if resolved is None:
                return "No files on this timeline yet — nothing to view."
            return self._view(resolved)
        if action == "create":
            if content is None or not filename:
                return "Error: 'create' needs both 'content' and a 'filename'."
            return self._create(target, content, filename, description)
        return f"Error: unknown action {action!r}. Use 'list', 'read', 'view', or 'create'."

    # --- uuid resolution -----------------------------------------------------

    def _resolve_uuid(self, uuid: str, timeline: str) -> str | None:
        """Resolve a `read`/`view` uuid, mapping the ``'latest'`` alias to the newest asset.

        An agent that just generated and posted an image cannot, on the same turn, view it
        without being handed the new asset's uuid — its own post is self-filtered from the
        wake's perception path, so the uuid never reaches its context (issue #161). The
        ``'latest'`` alias closes that gap: it resolves to the **most recent file on the
        target timeline** — which, right after a `generate_image`/`create`, is exactly the
        file the agent just posted. The SDK's asset filter is newest-first and lazily
        paginated, so this fetches only the first page's first item. Returns ``None`` when
        the timeline has no files at all (the caller turns that into a clean message); any
        other value is passed straight through as an explicit uuid.
        """
        if uuid.strip().lower() != "latest":
            return uuid
        newest = next(iter(self.context.client.assets.filter(timeline=timeline)), None)
        return newest.content.uuid if newest is not None else None

    # --- list ----------------------------------------------------------------

    def _list(self, timeline: str) -> str:
        client = self.context.client
        # Pull one past the cap so "there may be more" is only said when a
        # (DEFAULT_LIST_LIMIT + 1)th asset actually exists — never on an exact 50.
        # The SDK filter is lazy and paginating, so islice fetches only what it needs.
        assets = list(
            itertools.islice(client.assets.filter(timeline=timeline), DEFAULT_LIST_LIMIT + 1)
        )
        if not assets:
            return "No files on this timeline yet."
        lines = [_describe(asset) for asset in assets[:DEFAULT_LIST_LIMIT]]
        if len(assets) > DEFAULT_LIST_LIMIT:
            lines.append(f"(showing the {DEFAULT_LIST_LIMIT} most recent; there may be more)")
        return "Files on this timeline (newest first):\n" + "\n".join(lines)

    # --- read ----------------------------------------------------------------

    def _read(self, uuid: str) -> str:
        client = self.context.client
        asset = client.assets.get(uuid)
        file = asset.content.file
        meta = _describe(asset)

        if not _is_text(file.content_type) or file.byte_size > MAX_INLINE_BYTES:
            why = (
                "binary"
                if not _is_text(file.content_type)
                else f"{file.byte_size} bytes, over the {MAX_INLINE_BYTES}-byte inline limit"
            )
            if _is_image(file.content_type):
                hint = " Use action='view' to look at it."
            elif _is_audio(file.content_type):
                hint = " Use the 'listen' tool to hear what it says."
            else:
                hint = ""
            return f"{meta}\n({why} — not inlined. The file is on the timeline by that uuid.{hint})"

        data = _download(file.url)
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            # Declared text but not valid UTF-8 — treat as binary rather than guess.
            return f"{meta}\n(not valid UTF-8 text — not inlined.)"
        return f"{meta}\n\n{text}"

    # --- view ----------------------------------------------------------------

    def _view(self, uuid: str) -> str | ToolResult:
        """Fetch an image asset and hand it back for the model to actually see.

        Returns a `ToolResult` whose `images` the engine routes into the model's
        input as a data URL — self-contained, so it does not depend on the model's
        servers reaching the blob URL. A non-image (or oversized) asset comes back
        as a plain string explaining why it can't be shown, never as raw bytes.
        """
        asset = self.context.client.assets.get(uuid)
        meta = _describe(asset)
        result = image_input(asset.content.file)
        if isinstance(result, str):
            return f"{meta}\n({result})"  # a reason it can't be shown — not raw bytes
        return ToolResult(text=f"{meta}\nLooking at this image now.", images=[result])

    # --- create --------------------------------------------------------------

    def _create(self, timeline: str, content: str, filename: str, description: str | None) -> str:
        # The produced text goes straight to the upload as bytes — no temp file.
        asset = _upload(
            self.context.client, timeline, content.encode("utf-8"), filename, description
        )
        return f"Uploaded {filename!r} ({asset.content.file.byte_size} bytes). {_describe(asset)}"


# --- shared rendering / type helpers -----------------------------------------


def _download(url: str) -> bytes:
    """Fetch the blob bytes from its dereferenceable URL.

    A plain GET (following the redirect the platform's blob URL issues) — the URL is
    already authorized, so it carries no API token. Kept off the SDK on purpose: the
    SDK speaks to API paths, this is a direct blob fetch. Shared by the assets tool
    (`read`/`view`) and the audio tool (`listen`), the one place a blob is fetched.
    """
    response = httpx.get(url, follow_redirects=True, timeout=_DOWNLOAD_TIMEOUT)
    response.raise_for_status()
    return response.content


def _upload(client: Any, timeline: str, data: bytes, filename: str, description: str | None) -> Any:
    """Upload raw bytes as a named asset on a timeline; return the created asset.

    The one place the SDK upload contract lives, shared by the assets tool's
    ``create`` and the image generator: an in-memory buffer named so the SDK (and
    the server) see the filename, streamed straight to the multipart create. No
    temp file — the strongest version of "keep scratch bounded" is no scratch.
    """
    buffer = io.BytesIO(data)
    buffer.name = filename
    return client.timelines.get(timeline).assets.create(file=buffer, description=description)


def _describe(asset: Any) -> str:
    """One asset as a compact line: uuid, filename, size, type, and description.

    `description` is the one optional field — an asset uploaded without one. The
    SDK raises `AttributeError` (never `None`) for a field the API omitted, so it
    is read through `getattr` rather than assumed present; the file metadata
    (filename/size/type) is always there for a real blob.
    """
    file = asset.content.file
    line = (
        f"uuid={asset.content.uuid} · filename={file.filename!r} · "
        f"{file.byte_size} bytes · {file.content_type}"
    )
    description = getattr(asset.content, "description", None)
    if description:
        line += f" — {description}"
    return line


def _media_type(content_type: str) -> str:
    """The bare media type: parameters stripped, lowercased (e.g. ``image/png``).

    The one parser the type checks share, so ``view``'s viewable-type gate and the
    data URL it builds can never disagree on the same input.
    """
    return content_type.split(";", 1)[0].strip().lower()


def _is_text(content_type: str) -> bool:
    """Whether a file of this content type should be decoded and inlined as text."""
    if not content_type:
        return False
    base = _media_type(content_type)
    if base.startswith("text/"):
        return True
    if base.endswith("+json") or base.endswith("+xml"):
        return True
    return base in _TEXTUAL_APPLICATION_TYPES


def _is_image(content_type: str) -> bool:
    """Whether a file of this content type is an image (a candidate for `view`)."""
    if not content_type:
        return False
    return _media_type(content_type).startswith("image/")


def _is_audio(content_type: str) -> bool:
    """Whether a file of this content type is audio (a candidate for `listen`)."""
    if not content_type:
        return False
    return _media_type(content_type).startswith("audio/")


def image_input(file: Any) -> ImageContent | str:
    """A viewable image file as self-contained model input, or a reason it can't be shown.

    The single viewability gate shared by the assets ``view`` action and the asset-wake
    perception path (`_wake._perceive_asset`), so "the agent chose to look" and "the agent
    was shown a peer's file on wake" never diverge on *which* images render. Returns an
    `ImageContent` — the bytes fetched and inlined as a ``data:`` URL, so the input does
    not depend on the model's servers reaching the (possibly short-lived) blob URL — when
    the file is a supported image within the size ceiling. Otherwise returns a short reason
    string the caller surfaces. A download failure propagates as an ``httpx`` error for the
    caller to handle (the wake degrades to a description; ``view`` lets it surface).
    """
    if not _is_image(file.content_type):
        return "not an image — 'view' is for images. Use 'read' for text files."
    if _media_type(file.content_type) not in _VIEWABLE_IMAGE_TYPES:
        return "image type not viewable; supported: PNG, JPEG, GIF, WebP."
    if file.byte_size <= 0:
        return "empty file — nothing to view."
    if file.byte_size > MAX_IMAGE_BYTES:
        return f"{file.byte_size} bytes, over the {MAX_IMAGE_BYTES}-byte view limit — too large to look at."
    data = _download(file.url)
    return ImageContent(url=_data_url(file.content_type, data), alt=file.filename)


def _data_url(content_type: str, data: bytes) -> str:
    """A ``data:`` URL for the image bytes, so the input is self-contained.

    Uses the bare media type (stripped of any parameters) and base64-encodes the
    blob — the form the model providers accept for an inline image.
    """
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{_media_type(content_type)};base64,{encoded}"
