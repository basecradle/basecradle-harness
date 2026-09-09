"""Give the agent files — and senses for them: `assets` is the noun, the verb is the sense.

This is the first tool that acts *on* the platform, so it is the first
`PlatformTool` — it reaches the SDK client and the current timeline through the
bound `PlatformContext` (see `_platform.py`). Everything else is the same small
contract `MemoryTool` follows: an `action` enum, branching in `run`, a string
back for the model to read. A contributor adds the next platform tranche (tasks,
participants, …) by copying this shape.

The actions, the file equivalent of what a human peer does on a timeline:

- **list** — what files are here, with the uuids needed to open them.
- **read** — download one file and surface it. The model is text, so a text-ish
  file comes back decoded; a binary (or oversized) file comes back as metadata
  plus a "not inlined" note rather than a wall of bytes dumped into context.
- **view** — *look at* an image file. Where `read` refuses a binary, `view`
  fetches an image and hands it back as a `ToolResult` carrying the picture, so a
  vision-capable model actually sees it. This is the on-demand "look at this
  asset" step — images are never inlined eagerly, only when the agent chooses to look.
- **watch** — *watch* a video file, the same shape one tier down (`_video.py`).
- **listen** — *hear* an audio file, transcribed to text (`_audio.py`).
- **create** — upload content the agent produced (the common path: text → file),
  with an optional description.
- **post_image** — put an image another tool returned (a browser screenshot) on the timeline.

One noun, three senses — the founder's ruling, 2026-09-09 (issue #484)
-----------------------------------------------------------------------
`view` was an action here, video was a standalone ``watch_video`` tool, and audio a standalone
``hear_audio`` tool. That split was implementation history, not design: all three are *the agent
opening a file that is already on this timeline*, and they now spell it one way. The two standalone
tools are retired, their plugin files removed, and an upgrade prunes a stale copy out of an
existing overlay so a dead tool can never be resurrected from disk.

**Don't show a locked door.** The action list is built when the tool is constructed, from what the
agent is actually configured for — so `listen`, the one sense that needs a provider call, is absent
from the schema *and* from the description on an agent with no transcription provider, rather than
present and failing. `assets_options` is where that is decided, from the same
`ActivationContext` every other activation gate reads, and any future gated verb goes through it.

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
import logging
from typing import TYPE_CHECKING, Any

import httpx

from basecradle_harness._exceptions import ProviderConnectionError, ProviderError
from basecradle_harness._idempotency import ASSET
from basecradle_harness._messages import FrameSampling, ImageContent, ToolResult, VideoContent
from basecradle_harness._platform import PlatformTool

if TYPE_CHECKING:  # type-only: neither is a runtime dependency of this module
    # `_audio` reaches the model provider, which reaches back here — so the transcriber is
    # imported at the one call that needs it (`_listen`), never at module scope.
    from basecradle_harness._audio import Transcriber
    from basecradle_harness._plugins import ActivationContext

_log = logging.getLogger("basecradle_harness")

# A text-ish file is decoded and inlined; anything else is described, not dumped.
# `byte_size` over this cap is treated as not-inlinable even when text, so a huge
# log never blows up the model's context. 256 KiB is generous for prose/code.
MAX_INLINE_BYTES = 256 * 1024

# The largest image `view`/perception will load and base64-inline as model input. This is a
# **machine sanity bound — what this box will hold in memory — not a prediction of any vendor's
# input ceiling** (issue #336, decision 2). The old rationale ("matches OpenAI's per-image input
# ceiling") was the same disease as a vendor cap table: a local guess at a remote limit that rots the
# day a vendor changes it, and — worse — *pre-empts the vendor's own verdict*. So the number now
# bounds only the one genuinely-local fact, the RAM cost of loading the blob and its base64/data-URL
# expansion (roughly 2–3× the raw size, transiently). Everything under it is sent to the vendor for a
# verdict; the vendor's own rejection of an over-large payload is now reported to the timeline
# verbatim (the failure taxonomy — `ProviderPayloadTooLargeError`), so the harness no longer needs to
# second-guess it here. A file over *this* bound is still described, not shown, and that refusal stays
# a visible tool-result / perception string on every path (`image_input`, `_wake._perceive_asset`).
# The original bytes are never modified anywhere — no downscaling, no recompression (decision 1).
MAX_IMAGE_BYTES = 64 * 1024 * 1024

# Image content types a vision model can take as input. Kept to the formats the
# model providers document so `view` gives a clean "can't show that" rather than
# letting an unsupported type fail deep in the provider call.
_VIEWABLE_IMAGE_TYPES = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})

# The largest video the assets tool's `watch` will load. Exactly the same *kind* of number as
# `MAX_IMAGE_BYTES` above and for exactly the same reason: a **machine RAM sanity bound — what
# this box will hold in memory — never a prediction of any vendor's input ceiling** (issue #336,
# decision 2; the vendor's own rejection is relayed verbatim, so the harness has no business
# guessing it). A clip is held three ways at once, transiently: the downloaded bytes, the base64
# data URL (~1.33x), and, on the frames path, one decoded frame at a time — the decode is
# streaming, so the frame count never enters the bound. A clip *at* the bound therefore costs
# roughly 600 MB transiently, which is the figure to weigh when sizing a fleet box, and it is the
# reason this is a knob-shaped constant rather than a number nobody wrote down. Over the bound the
# asset is described, not fetched, with the same wording pattern an oversized image gets. The
# original bytes are never modified *to fit a limit* — no transcoding, no re-encoding, here or on
# any perception path (decision 1).
#
# **One thing does cut a clip's bytes, and it falls outside decision 1 because of who asked**
# (issue #482, founder-decided 2026-09-09): the assets tool's `watch` narrows a clip to the window
# **the agent itself requested**, in memory, for that one turn (`_video.native_watch`). Decision 1
# is about the harness quietly reshaping a file to satisfy a *vendor*; a window the model asked for
# in its own tool call is the tool doing what it was told, and nothing on the timeline changes —
# the Asset is untouched and the cut is evicted with the rest of the turn's payload.
MAX_VIDEO_BYTES = 256 * 1024 * 1024

# Video content types a video-capable model can take as input, and that PyAV can decode for the
# frames fallback. Kept to the container formats the providers document, so `watch` gives a
# clean "can't show that" rather than failing deep in a decoder or a provider call.
_VIEWABLE_VIDEO_TYPES = frozenset({"video/mp4", "video/webm", "video/quicktime", "video/mpeg"})

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


#: Every action the assets tool can carry, in the order the model reads them: the file verbs, the
#: three senses, then the two that put something *on* the timeline. The actual set a given agent
#: sees is this filtered by `assets_options` — see `AssetsTool.__init__`.
_ALL_ACTIONS = ("list", "read", "view", "watch", "listen", "create", "post_image")

#: The sense actions that open a file rather than change the timeline. `read` joins them for the
#: ``uuid`` parameter's wording, which is the one place all four are named together.
_UUID_ACTIONS = ("read", "view", "watch", "listen")

#: One clause per action, for the description built at construction. Kept as data rather than a
#: paragraph so a configuration-dependent action can be dropped without leaving a dangling
#: half-sentence behind it — the "don't show a locked door" rule applies to the prose the model
#: reads as much as to the enum it calls.
_ACTION_TEXT = {
    "list": "action='list' shows the files here with their uuids",
    "read": (
        "action='read' downloads one file by uuid and returns its text (binary files come back "
        "as a description, not raw bytes)"
    ),
    "view": (
        "action='view' looks at an image file by uuid so you can actually see it and describe or "
        "reason about it"
    ),
    "watch": (
        "action='watch' watches a video file by uuid — optional 'every' sets the seconds between "
        "the frames you are shown (default 1), and 'start'/'end' (seconds) narrow the window so "
        "you can look closely at one moment of a long clip"
    ),
    "listen": (
        "action='listen' hears an audio file by uuid and returns a transcript of what was said"
    ),
    "create": (
        "action='create' uploads a new file from the text content you provide, with a filename "
        "and an optional description"
    ),
    "post_image": (
        "action='post_image' posts an image a tool just returned (such as a browser screenshot "
        "from an MCP tool) to the timeline — pass its reference in 'image' (e.g. 'mcp-image-1', "
        "shown in the tool result, or 'latest' for the most recent), so you can share what you "
        "captured even if you cannot see it yourself"
    ),
}


def assets_options(ctx: ActivationContext) -> dict[str, Any]:
    """The assets tool's constructor options for one config — which senses this agent has.

    The `ToolPlugin.configure` hook the shipped ``tools/assets.py`` names (issue #484). It answers
    the one question the tool cannot: `view` and `watch` cost no provider call and are therefore
    always there, while `listen` needs a transcription provider — so on an agent without one the
    action is absent from the schema and the description rather than present and failing. That is
    the "don't show a locked door" rule, and any future gated verb is decided here beside this one.

    The gate is `OpenAIKey`, **the same requirement object** the retired ``hear_audio`` plugin
    declared, evaluated against the same `ActivationContext` — never a second reading of the
    environment that could drift from it. It is imported at the call rather than at module scope so
    the tool module does not depend on the resolution machinery that configures it.
    """
    from basecradle_harness._plugins import OpenAIKey

    return {"listen": OpenAIKey().met(ctx)}


class AssetsTool(PlatformTool):
    """Open and exchange files (assets) on the agent's current timeline — and sense them.

    A `PlatformTool`: the hosting agent binds the SDK client and current-timeline
    uuid before the loop runs. Until bound, `run` reports it is not connected
    (via `PlatformError`) rather than failing obscurely.

    Args:
        listen: Whether this agent has a transcription provider, and so whether the ``listen``
            action exists at all. The plugin path answers it from the active config
            (`assets_options`); a library caller passes it directly.
        transcriber: The transcription collaborator (`_audio.Transcriber`) — pass one to tune the
            model, base URL or timeout. Passing one implies `listen`, so a caller never has to say
            the same thing twice.
    """

    name = "assets"

    def __init__(self, *, listen: bool = False, transcriber: Transcriber | None = None) -> None:
        self._transcriber = transcriber
        can_hear = listen or transcriber is not None
        #: The actions this instance actually offers, in `_ALL_ACTIONS` order — and the **single**
        #: answer to "does this agent have that sense?", read by `run`, by `_read`'s hint and by
        #: `_unknown`. Instance state, not class state: two agents in one process (a test, a future
        #: sub-agent) must be able to have different senses, and `Tool.to_spec` reads
        #: `self.description`/`self.parameters`, which are built from this.
        self.actions = tuple(a for a in _ALL_ACTIONS if a != "listen" or can_hear)
        self.description = _description(self.actions)
        self.parameters = _parameters(self.actions)

    def run(
        self,
        action: str,
        uuid: str | None = None,
        content: str | None = None,
        filename: str | None = None,
        description: str | None = None,
        timeline: str | None = None,
        image: str | None = None,
        every: float | None = None,
        start: float | None = None,
        end: float | None = None,
    ) -> str | ToolResult:
        """Dispatch on `action`. Returns a message written for the model to read.

        `view` and `watch` may return a `ToolResult` carrying the picture or the clip for the model
        to perceive; every other action returns a plain string.
        """
        # Validate before reaching for the platform: an unknown action and a missing uuid are
        # answerable without a client, and an unbound tool should say what is actually wrong.
        if action not in self.actions:
            return _unknown(action, self.actions)
        if action in _UUID_ACTIONS and (not uuid or not uuid.strip()):
            return f"Error: {action!r} needs the asset's uuid. Use 'list' to find it."

        target = timeline or self.context.timeline
        if action == "list":
            return self._list(target)
        if action in _UUID_ACTIONS:
            assert uuid is not None  # guaranteed above
            resolved = self._resolve_uuid(uuid, target)
            if resolved is None:
                return f"No files on this timeline yet — nothing to {action}."
            if action == "read":
                return self._read(resolved)
            if action == "view":
                return self._view(resolved)
            if action == "watch":
                return self._watch(resolved, every, start, end)
            return self._listen(resolved)
        if action == "create":
            # Minted before the validation, never after: the ordinal is counted off the
            # transcript, which records this call either way (issue #297 — see `PlatformTool.key`).
            key = self.key(ASSET)
            if content is None or not filename:
                return "Error: 'create' needs both 'content' and a 'filename'."
            return self._create(target, content, filename, description, key)
        return self._post_image(target, image, filename, description)

    # --- uuid resolution -----------------------------------------------------

    def _resolve_uuid(self, uuid: str, timeline: str) -> str | None:
        """This tool's binding of the shared `resolve_uuid` (the ``'latest'`` alias)."""
        return resolve_uuid(self.context.client, uuid, timeline)

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
            # The hint names the sense that opens *this* kind of file — and only when this agent
            # actually has it, so a `listen`-less agent is never pointed at an action that is not
            # in its schema (the "don't show a locked door" rule, issue #484).
            if _is_image(file.content_type):
                hint = " Use action='view' to look at it."
            elif _is_video(file.content_type):
                hint = " Use action='watch' to watch it."
            elif _is_audio(file.content_type) and "listen" in self.actions:
                hint = " Use action='listen' to hear what it says."
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

        The tool result carries only the asset's metadata (`_describe` — filename,
        type, size, and any description); it does **not** narrate perception. Whether
        the pixels are actually *seen* is the engine's call, not the tool's: a tool has
        no view of the provider (the body/brain split — `PlatformContext` is client +
        timeline only), and only the engine knows whether the model can take an image.
        So the engine gates the pixels on `model_sees_images` and captions them (or, for
        a text-only model, says plainly they were described, not shown — issue #316).
        Claiming "looking at it now" here would promise vision the tool cannot verify.
        """
        asset = self.context.client.assets.get(uuid)
        meta = _describe(asset)
        result = image_input(asset.content.file)
        if isinstance(result, str):
            return f"{meta}\n({result})"  # a reason it can't be shown — not raw bytes
        return ToolResult(text=meta, images=[result])

    # --- watch ---------------------------------------------------------------

    def _watch(
        self, uuid: str, every: float | None, start: float | None, end: float | None
    ) -> str | ToolResult:
        """Fetch a video asset and hand it back for the model to actually watch.

        `view`'s video sibling, and the same division of labour: this fetches, and the **engine**
        decides what the model receives — the clip itself (trimmed to `start`/`end`), sampled
        frames, or an honest caption — from the provider's own declared capabilities
        (`_engine._show_video`). A tool has no view of the provider, so it narrates no perception
        (issue #316).

        There is **no provider call and no idempotency key**: nothing is created, nothing is spent,
        nothing is posted. The clip is decoded in-process by PyAV, whose wheels bundle FFmpeg, so
        this stays inside the locked profile's no-shell boundary.

        `every`/`start`/`end` ride along on the returned clip (`FrameSampling`) rather than being
        acted on here, because the tier that honors them is chosen later, in the engine, by which
        time the tool that knew what the agent asked for is long gone.
        """
        from basecradle_harness._video import DEFAULT_EVERY, clip_facts

        asset = self.context.client.assets.get(uuid)
        meta = _describe(asset)
        sampling = FrameSampling(
            every=every if every and every > 0 else DEFAULT_EVERY, start=start, end=end
        )
        result = video_input(asset.content.file, sampling)
        if isinstance(result, str):
            return f"{meta}\n({result})"  # a reason it can't be watched — never raw bytes
        return ToolResult(text=f"{meta}\n({clip_facts(result)})", videos=[result])

    # --- listen --------------------------------------------------------------

    def _listen(self, uuid: str) -> str:
        """Fetch an audio asset, transcribe it, and return the transcript for the model to read.

        The one sense that costs a **provider call** (`_audio.Transcriber`), which is why it is the
        one action an agent may not have at all — an agent with no transcription provider never
        sees it in the schema, so reaching this method means the capability is configured.

        The wrong kind of file (and an empty or oversized one) is refused *before* downloading or
        calling the provider — the same discipline `view` and `watch` follow.
        """
        from basecradle_harness._audio import MAX_AUDIO_BYTES, Transcriber

        asset = self.context.client.assets.get(uuid)
        file = asset.content.file
        meta = _describe(asset)

        if not _is_audio(file.content_type):
            return (
                f"{meta}\n(not an audio file — 'listen' is for audio. Use 'read' for text, "
                "'view' for images, 'watch' for video.)"
            )
        if file.byte_size <= 0:
            return f"{meta}\n(empty file — nothing to hear.)"
        if file.byte_size > MAX_AUDIO_BYTES:
            return (
                f"{meta}\n({file.byte_size} bytes, over the {MAX_AUDIO_BYTES}-byte "
                "transcription limit — too large to listen to.)"
            )

        transcriber = self._transcriber or Transcriber()
        key = transcriber.key
        if not key:
            return (
                "Error: no API key for transcription. Set AI_API_KEY "
                "(or pass a configured Transcriber to AssetsTool)."
            )

        data = _download(file.url)
        try:
            transcript = transcriber.transcribe(data, file.filename, file.content_type, key)
        except ProviderConnectionError as exc:
            return f"Error transcribing audio: could not reach the transcription API: {exc}"
        except ProviderError as exc:
            return f"Error transcribing audio: {exc}"

        if not transcript.strip():
            return f"{meta}\n(transcribed, but no speech was detected.)"
        return f"{meta}\n\nTranscript:\n{transcript}"

    # --- create --------------------------------------------------------------

    def _create(
        self,
        timeline: str,
        content: str,
        filename: str,
        description: str | None,
        key: str | None = None,
    ) -> str:
        """Upload the model's text as an Asset. `key` is the deterministic Idempotency-Key (#297).

        The content is inline in the tool call, so the re-issue a resume makes after a killed wake
        replays byte-identical bytes — and the key means that even if the dead wake's upload landed,
        the platform hands back the original Asset rather than storing the file twice.
        """
        # The produced text goes straight to the upload as bytes — no temp file.
        asset = _upload(
            self.context.client,
            timeline,
            content.encode("utf-8"),
            filename,
            description,
            idempotency_key=key,
        )
        self.acted("asset", asset.content.uuid)  # a file on a timeline is a visible act (#293)
        return f"Uploaded {filename!r} ({asset.content.file.byte_size} bytes). {_describe(asset)}"

    # --- post_image ----------------------------------------------------------

    def _post_image(
        self,
        timeline: str,
        image: str | None,
        filename: str | None,
        description: str | None,
    ) -> str:
        """Post an image an MCP tool returned (a browser screenshot) to the timeline (issue #318).

        The "show me what you see" path, and it is **independent of the model's vision**: the bytes
        were stashed in the per-wake `McpImageStore` when the tool returned them, so a text-only
        agent can share a screenshot it cannot itself see. `image` is the reference the tool result
        named (``mcp-image-1``), or ``'latest'`` / omitted for the most recent capture.

        Uploads through the same asset-create path as everything else — but with **no idempotency
        key**, exactly like a generated image's upload (`_upload`, `generate_image`). That is
        deliberate and load-bearing: the bytes live only in the volatile store, so a killed-and-
        resumed wake cannot reconstruct them, which means the create must never be re-issued by the
        recovery. Routing it through the keyed ``create`` action would classify it as a replayable
        create (`_idempotency.CREATE_CALLS`) and the resume would try — and fail — to replay bytes
        that no longer exist; a distinct action keeps it out of that set entirely.
        """
        store = self.context.mcp_images
        if store is None or len(store) == 0:
            return (
                "Error: no captured images to post. An image only becomes postable after a tool "
                "(e.g. a browser screenshot) returns one this wake; captures do not survive a "
                "restart."
            )
        ref = (image or "latest").strip()
        stashed = store.get(ref)
        if stashed is None:
            return (
                f"Error: no captured image {ref!r}. Use the reference shown in the tool result "
                "(e.g. 'mcp-image-1'), or 'latest' for the most recent capture."
            )
        name = filename or _capture_filename(stashed.mimetype)
        asset = _upload(self.context.client, timeline, stashed.data, name, description)
        self.acted("asset", asset.content.uuid)  # a file on a timeline is a visible act (#293)
        return (
            f"Posted {name!r} ({asset.content.file.byte_size} bytes) from capture {ref!r}. "
            f"{_describe(asset)}"
        )


# --- the configured action set: schema and description ------------------------


def _unknown(action: str, actions: tuple[str, ...]) -> str:
    """The error for an action this agent does not have — naming the ones it does.

    Reads off the *configured* set, so an agent with no transcription never suggests `listen` even
    when the model guessed it: an error message that offers a door that is not there is the same
    defect as a schema that does.
    """
    offered = ", ".join(repr(name) for name in actions)
    return f"Error: unknown action {action!r}. Use {offered}."


def _description(actions: tuple[str, ...]) -> str:
    """The model-facing description for one configured action set (issue #484).

    Built from `_ACTION_TEXT` rather than written as a paragraph so that dropping a gated action
    drops its clause with it. The senses are named together in one sentence at the end because
    *"this is how you check what you made"* is the thing an agent most often needs told, and it is
    exactly as true of a clip as of a picture.
    """
    clauses = "; ".join(_ACTION_TEXT[name] for name in actions)
    senses = [name for name in ("view", "watch", "listen") if name in actions]
    check = (
        f" Use {_join(senses)} to check media you generated yourself — open it and see whether it "
        "is what you asked for."
        if senses
        else ""
    )
    return (
        "Exchange files on the timeline, the way a peer shares an attachment, and open the ones "
        f"that are here. {clauses}. Anywhere a uuid is asked for you can pass 'latest' instead, "
        "for the most recent file on the timeline — such as a file you just posted, so you can "
        f"open it without being handed its uuid.{check} Operations use the current timeline "
        "unless you pass a timeline uuid. Assets are shared with every viewer and can never be "
        "edited or deleted — prefer your own storage for private or working files; upload what is "
        "meant for the peers here. Platform REST: POST /timelines/{timeline_uuid}/assets — this "
        "tool calls that same endpoint; "
        "https://basecradle.com/docs/api.md#tools-and-the-http-api has the full API."
    )


def _parameters(actions: tuple[str, ...]) -> dict[str, Any]:
    """The JSON-Schema parameters for one configured action set.

    Only the ``action`` enum and the ``uuid`` wording vary today (the gated verb is `listen`), but
    the whole schema is built here rather than patched at one key, so a future action carrying a
    parameter of its own has one obvious place to add it.
    """
    opens = _join([f"'{name}'" for name in _UUID_ACTIONS if name in actions])
    return {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": list(actions),
                "description": "What to do.",
            },
            "uuid": {
                "type": "string",
                "description": (
                    f"The asset's uuid ({opens}). Get it from 'list', or pass "
                    "'latest' for the most recent file on the timeline — e.g. an image "
                    "you just generated and posted, so you can open it without being "
                    "handed its uuid."
                ),
            },
            "every": {
                "type": "number",
                "description": (
                    "Optional seconds between the frames you are shown (watch only, default 1). "
                    "Smaller sees more motion; a per-call frame cap still applies, so narrow the "
                    "window with 'start'/'end' to actually look closer."
                ),
            },
            "start": {
                "type": "number",
                "description": (
                    "Optional start of the window to watch, in seconds from the clip's start "
                    "(watch only)."
                ),
            },
            "end": {
                "type": "number",
                "description": (
                    "Optional end of the window to watch, in seconds from the clip's start "
                    "(watch only)."
                ),
            },
            "image": {
                "type": "string",
                "description": (
                    "The reference of a captured image to post (post_image only) — e.g. "
                    "'mcp-image-1' as shown in a tool result, or 'latest' for the most "
                    "recently captured image. Defaults to the latest capture if omitted."
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


def _join(items: list[str]) -> str:
    """``a``, ``a and b``, ``a, b and c`` — for a clause whose length depends on the config."""
    if len(items) <= 1:
        return "".join(items)
    return f"{', '.join(items[:-1])} and {items[-1]}"


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


def _upload(
    client: Any,
    timeline: str,
    data: bytes,
    filename: str,
    description: str | None,
    *,
    idempotency_key: str | None = None,
) -> Any:
    """Upload raw bytes as a named asset on a timeline; return the created asset.

    The one place the SDK upload contract lives, shared by the assets tool's
    ``create`` and the image generator: an in-memory buffer named so the SDK (and
    the server) see the filename, streamed straight to the multipart create. No
    temp file — the strongest version of "keep scratch bounded" is no scratch.

    `idempotency_key` is passed only by the **assets tool** (issue #297), and its absence
    everywhere else is deliberate rather than an oversight. A generated image's upload is not
    re-issued by a recovery — the *generation* that produced those bytes is the non-idempotent act,
    and no key can un-spend it, so a killed wake's `generate_image` is surfaced to the model rather
    than re-run. Keying the upload would buy nothing and cost the one thing that matters here:
    the ordinal count, which must see exactly the creates the transcript's tool calls describe.
    """
    buffer = io.BytesIO(data)
    buffer.name = filename
    return client.timelines.get(timeline).assets.create(
        file=buffer, description=description, idempotency_key=idempotency_key
    )


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


# The extension a `post_image` capture is named with when the model gives no filename,
# by media type. An unknown type falls back to ``.img`` — the bytes upload either way.
_CAPTURE_EXTENSIONS = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/gif": "gif",
    "image/webp": "webp",
}


def _capture_filename(content_type: str) -> str:
    """A default filename for a posted capture (``capture.png``) from its media type."""
    return f"capture.{_CAPTURE_EXTENSIONS.get(_media_type(content_type), 'img')}"


def _is_text(content_type: str) -> bool:
    """Whether a file of this content type should be decoded and inlined as text."""
    if not content_type:
        return False
    base = _media_type(content_type)
    if base.startswith("text/"):
        return True
    if base.endswith(("+json", "+xml")):
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


def _is_video(content_type: str) -> bool:
    """Whether a file of this content type is video (a candidate for `watch`)."""
    if not content_type:
        return False
    return _media_type(content_type).startswith("video/")


def resolve_uuid(client: Any, uuid: str, timeline: str) -> str | None:
    """Resolve an asset uuid, mapping the ``'latest'`` alias to the newest file on the timeline.

    An agent that just generated and posted a file cannot, on the same turn, open it without
    being handed the new asset's uuid — its own post is self-filtered from the wake's perception
    path, so the uuid never reaches its context (issue #161). The ``'latest'`` alias closes that
    gap: it resolves to the **most recent file on the target timeline** — which, right after a
    `generate_image` / `grok_generate_video` / `create`, is exactly the file the agent just
    posted. The SDK's asset filter is newest-first and lazily paginated, so this fetches only the
    first page's first item. Returns ``None`` when the timeline has no files at all (the caller
    turns that into a clean message); any other value is passed straight through as an explicit
    uuid.

    Shared by every uuid-taking assets action — `read`, `view`, `watch`, `listen` — so an agent
    that can say ``'latest'`` to one can say it to all of them; a clip is the case that needs it
    most, since watching what it just generated is how the agent checks its own work.
    """
    if uuid.strip().lower() != "latest":
        return uuid
    newest = next(iter(client.assets.filter(timeline=timeline)), None)
    return newest.content.uuid if newest is not None else None


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


def model_sees_images(provider: object) -> bool:
    """Whether the configured model can take image input — the asset-wake's vision gate (#228).

    Companion to `image_input`: that answers *"is this file viewable?"*, this answers *"can this
    model view?"*. It reads the optional `supports_vision` provider capability (only the OpenRouter
    adapter answers it today, from the model's ``architecture.input_modalities``) and is deliberately
    **fail-open**: an absent capability, a ``None`` (unknown) answer, or a raise all read as "assume
    it can see," so the *only* behavior change is for a model that **definitely** reports no vision.

    That default is the safe one. Every model the fleet runs on an image-serializing surface is
    vision-capable, so withholding an image on a wrong guess is a real regression, while showing one
    to a text-only model is what this gate exists to catch through the *definite* ``False`` — the one
    case that is both knowable and worth acting on.
    """
    capability = getattr(provider, "supports_vision", None)
    if not callable(capability):
        return True
    try:
        answer = capability()
    except Exception as exc:  # noqa: BLE001 - a metadata read must never break a wake
        _log.warning("Could not read the model's vision capability from the provider: %s", exc)
        return True
    return answer is not False  # True/None/unknown → show the image; only a definite False degrades


def video_input(file: Any, sampling: FrameSampling | None = None) -> VideoContent | str:
    """A watchable video file as self-contained model input, or a reason it can't be shown.

    `image_input`'s video sibling, and deliberately the same shape: a type gate, a size gate, one
    download, and the bytes inlined as a ``data:`` URL so the input never depends on a vendor's
    servers reaching a short-lived, access-controlled blob URL. Returns a `VideoContent` on
    success, otherwise a short reason string the caller surfaces to the model.

    The `VideoContent` is what the **engine** then routes by capability — natively to a model that
    takes video, as sampled frames to one that takes only images, as an honest caption to one that
    takes neither. This function does not know or care which; that is the engine's call, exactly as
    it is for an image (`_engine._show_media`).

    `sampling` is the agent's own request for how to look — every N seconds, over an optional
    window — carried along so the frames tier can honor it. It rides on the `VideoContent` rather
    than being passed separately because the tier is chosen *later*, in the engine, by which time
    the tool that knew what the agent asked for is long gone.

    A download failure propagates as an ``httpx`` error for the caller to handle.
    """
    if not _is_video(file.content_type):
        return (
            "not a video — 'watch' is for videos. Use 'view' for an image and 'read' for a "
            "text file."
        )
    if _media_type(file.content_type) not in _VIEWABLE_VIDEO_TYPES:
        return "video type not watchable; supported: MP4, WebM, QuickTime, MPEG."
    if file.byte_size <= 0:
        return "empty file — nothing to watch."
    if file.byte_size > MAX_VIDEO_BYTES:
        return (
            f"{file.byte_size} bytes, over the {MAX_VIDEO_BYTES}-byte watch limit — "
            "too large to load."
        )
    data = _download(file.url)
    return VideoContent(
        url=_data_url(file.content_type, data),
        alt=file.filename,
        content_type=_media_type(file.content_type),
        sampling=sampling or FrameSampling(),
    )


def model_sees_video(provider: object) -> bool:
    """Whether the configured model can take **video** input, read from `supports_video` (#471).

    `model_sees_images`' sibling with the **opposite default, on purpose**, and the asymmetry is
    the whole design — so it is stated here rather than left to be rediscovered:

    - `model_sees_images` **fails open**. There is no fallback below an image: withholding one on
      a wrong guess is a real regression, and every model the fleet runs on an image-serializing
      surface can see. Only a *definite* ``False`` degrades.
    - `model_sees_video` **fails closed**. There **is** a fallback below video — sampled frames,
      which every vision model can take — so a wrong guess costs a tier, not the capability. And
      the two errors are not symmetric: a video part sent to a model that cannot take one is a
      hard 400 that fails the whole wake, where guessing low merely samples frames the model could
      have watched natively. Guess toward the outcome that still works.

    So only a capability that answers a definite ``True`` sends native video; absent, ``None``,
    or a raise all mean frames.
    """
    capability = getattr(provider, "supports_video", None)
    if not callable(capability):
        return False
    try:
        answer = capability()
    except Exception as exc:  # noqa: BLE001 - a metadata read must never break a wake
        _log.warning("Could not read the model's video capability from the provider: %s", exc)
        return False
    return answer is True  # only a definite yes; everything else falls back to frames


def _data_url(content_type: str, data: bytes) -> str:
    """A ``data:`` URL for the image bytes, so the input is self-contained.

    Uses the bare media type (stripped of any parameters) and base64-encodes the
    blob — the form the model providers accept for an inline image.
    """
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{_media_type(content_type)};base64,{encoded}"
