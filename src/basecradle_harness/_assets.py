"""Give the agent files: list, read, and create assets on a BaseCradle timeline.

This is the first tool that acts *on* the platform, so it is the first
`PlatformTool` — it reaches the SDK client and the current timeline through the
bound `PlatformContext` (see `_platform.py`). Everything else is the same small
contract `MemoryTool` follows: an `action` enum, branching in `run`, a string
back for the model to read. A contributor adds the next platform tranche (tasks,
participants, …) by copying this shape.

Three actions, the file equivalent of what a human peer does on a timeline:

- **list** — what files are here, with the uuids needed to read them.
- **read** — download one file and surface it. The model is text, so a text-ish
  file comes back decoded; a binary (or oversized) file comes back as metadata
  plus a "not inlined" note rather than a wall of bytes dumped into context.
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

import io
import itertools
from typing import Any

import httpx

from basecradle_harness._platform import PlatformTool

# A text-ish file is decoded and inlined; anything else is described, not dumped.
# `byte_size` over this cap is treated as not-inlinable even when text, so a huge
# log never blows up the model's context. 256 KiB is generous for prose/code.
MAX_INLINE_BYTES = 256 * 1024

# How many assets one `list` returns. A timeline rarely has more; the cap keeps a
# pathological one from flooding context. When it bites, the reply says so.
DEFAULT_LIST_LIMIT = 50

# Download timeout (seconds) for a `read`. The URL is a dereferenceable blob link.
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
    """List, read, and create files (assets) on the agent's current timeline.

    A `PlatformTool`: the hosting agent binds the SDK client and current-timeline
    uuid before the loop runs. Until bound, `run` reports it is not connected
    (via `PlatformError`) rather than failing obscurely.
    """

    name = "assets"
    description = (
        "Exchange files on the timeline, the way a peer shares an attachment. "
        "action='list' shows the files here with their uuids; action='read' "
        "downloads one file by uuid and returns its text (binary files come back as "
        "a description, not raw bytes); action='create' uploads a new file from the "
        "text content you provide, with a filename and an optional description. "
        "Operations use the current timeline unless you pass a timeline uuid."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "read", "create"],
                "description": "What to do.",
            },
            "uuid": {
                "type": "string",
                "description": "The asset's uuid (read only). Get it from 'list'.",
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
    ) -> str:
        """Dispatch on `action`. Returns a message written for the model to read."""
        target = timeline or self.context.timeline
        if action == "list":
            return self._list(target)
        if action == "read":
            if not uuid:
                return "Error: 'read' needs the asset's uuid. Use 'list' to find it."
            return self._read(uuid)
        if action == "create":
            if content is None or not filename:
                return "Error: 'create' needs both 'content' and a 'filename'."
            return self._create(target, content, filename, description)
        return f"Error: unknown action {action!r}. Use 'list', 'read', or 'create'."

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
            return f"{meta}\n({why} — not inlined. The file is on the timeline by that uuid.)"

        data = self._download(file.url)
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            # Declared text but not valid UTF-8 — treat as binary rather than guess.
            return f"{meta}\n(not valid UTF-8 text — not inlined.)"
        return f"{meta}\n\n{text}"

    @staticmethod
    def _download(url: str) -> bytes:
        """Fetch the blob bytes from its dereferenceable URL.

        A plain GET (following the redirect the platform's blob URL issues) — the
        URL is already authorized, so it carries no API token. Kept off the SDK on
        purpose: the SDK speaks to API paths, this is a direct blob fetch.
        """
        response = httpx.get(url, follow_redirects=True, timeout=_DOWNLOAD_TIMEOUT)
        response.raise_for_status()
        return response.content

    # --- create --------------------------------------------------------------

    def _create(self, timeline: str, content: str, filename: str, description: str | None) -> str:
        client = self.context.client
        # An in-memory buffer named so the SDK (and the server) see the filename.
        # No temp file: the produced text goes straight to multipart upload.
        buffer = io.BytesIO(content.encode("utf-8"))
        buffer.name = filename
        assets = client.timelines.get(timeline).assets
        asset = assets.create(file=buffer, description=description)
        return f"Uploaded {filename!r} ({asset.content.file.byte_size} bytes). {_describe(asset)}"


# --- shared rendering / type helpers -----------------------------------------


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


def _is_text(content_type: str) -> bool:
    """Whether a file of this content type should be decoded and inlined as text."""
    if not content_type:
        return False
    base = content_type.split(";", 1)[0].strip().lower()
    if base.startswith("text/"):
        return True
    if base.endswith("+json") or base.endswith("+xml"):
        return True
    return base in _TEXTUAL_APPLICATION_TYPES
