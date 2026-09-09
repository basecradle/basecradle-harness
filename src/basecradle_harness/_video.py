"""Let the agent watch: fetch a video asset and put it in front of the model.

The video analog of the assets tool's `view` (see an image) and `HearAudioTool` (hear audio).
Until this, no harness agent could perceive video at all: `view` is images-only, a posted clip
on an `asset.created` wake was acknowledged in text and never seen, and an agent that generated
a video had no way to check its own work. On 2026-08-13 @eddie-murphy generated three clips for
@origin and asserted a first-frame match he had no way to verify — the founder's ruling is that
agents get eyes for video, as a **default tool for every harness agent**, and that the harness
never tells a model to ask a human to look (issue #471).

Three tiers, chosen by capability — never by vendor
----------------------------------------------------
The tool's job ends at *fetching the clip*; how the model perceives it is the **engine's** call,
exactly as it is for an image. `WatchVideoTool` returns a `VideoContent` and the engine routes it
(`_engine._show_media`):

1. **The model takes video** (`supports_video` answers a definite yes) → it gets the video.
2. **The model takes images** → `sample_frames` decodes frames here and the model gets those.
3. **The model takes neither** → an honest caption saying the clip was described, not shown.

This is one seam extended, not a second one built beside it: the same `_split_result` →
`_show_media` → evict path an image already travels. The tool has no view of the provider (the
body/brain split — a `PlatformContext` is client + timeline only), so it deliberately does not
narrate perception (issue #316): claiming "watching it now" would promise sight the tool cannot
verify.

Pure Python, no subprocess
--------------------------
Frames are decoded with **PyAV**, whose wheels bundle FFmpeg — so the decode happens *in this
process*, and the locked profile's no-shell boundary is untouched. No `ffmpeg` subprocess, no apt
package, nothing for `Policy.locked()` to have an opinion about. That is the whole reason
`watch_video` can be a benign default tool rather than a powerful opt-in one: seeing a file that
is already on the timeline costs no provider call and reaches nothing outside the process.

Frames exist **in memory only** — never written to disk, never posted as assets, and evicted
from the transcript after the turn that showed them, exactly like a viewed image.
"""

from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass
from typing import Any

from basecradle_harness._assets import _describe, resolve_uuid, video_input
from basecradle_harness._messages import FrameSampling, ImageContent, ToolResult, VideoContent
from basecradle_harness._platform import PlatformTool

#: The most frames one `watch_video` call will put in front of the model. A cap, not a knob: an
#: agent that wants a closer look narrows the **window** (``start``/``end``), which is both cheaper
#: and more informative than more frames over the same span. 24 frames at the default long edge is
#: a few hundred KB of JPEG — a real but bounded cost, paid once (they are evicted after the turn).
MAX_FRAMES = 24

#: Seconds between sampled frames, by default. One frame a second reads a short clip's motion
#: without paying for near-duplicates at 24 fps.
DEFAULT_EVERY = 1.0

#: The long edge each sampled frame is scaled down to, in pixels. Small enough that 24 of them are
#: affordable, large enough to read text and faces in a 720p/1080p clip. A frame *smaller* than
#: this is never scaled **up** — that would spend tokens on invented pixels.
FRAME_LONG_EDGE = 768

#: JPEG quality for a sampled frame. 80 is the usual "visually clean, meaningfully smaller" point;
#: a frame is evidence, not a deliverable, and the original clip is untouched on the timeline.
FRAME_QUALITY = 80


@dataclass
class VideoInfo:
    """What a probe read off a clip's header: how long, how fast, how big.

    `duration_s` and `fps` are ``0.0`` when the container states neither — a legitimately
    unknowable answer for some streams, and the sampler treats it as "decode and see" rather than
    guessing a number to print.
    """

    duration_s: float
    fps: float
    width: int
    height: int


def probe(video_bytes: bytes) -> VideoInfo:
    """Read a clip's duration, frame rate and resolution without decoding it.

    Header-only: PyAV answers all four off the container and stream metadata, so this costs a
    parse rather than a decode. Raises `ValueError` when the bytes are not a decodable video —
    the uniform parse-boundary failure this repo uses, which every caller turns into a readable
    line for the model rather than a traceback.
    """
    with _open(video_bytes) as container:
        stream = _video_stream(container)
        duration = _duration_of(container, stream)
        rate = stream.average_rate or stream.guessed_rate
        return VideoInfo(
            duration_s=duration,
            fps=float(rate) if rate else 0.0,
            width=int(stream.width or 0),
            height=int(stream.height or 0),
        )


def sample_frames(
    video_bytes: bytes,
    *,
    name: str = "video",
    every: float = DEFAULT_EVERY,
    start: float | None = None,
    end: float | None = None,
    max_frames: int = MAX_FRAMES,
    long_edge: int = FRAME_LONG_EDGE,
) -> tuple[str, list[ImageContent]]:
    """Decode a clip into a handful of stills a vision model can read, plus a summary of what was
    sampled.

    The tier-2 fallback: a model that takes images but not video sees the clip as frames. Targets
    are the **first frame of the window, then one every `every` seconds, then the last** — so the
    beginning and the end are always represented, which is what a "does frame 0 match the source
    still?" question actually needs.

    Three properties are deliberate:

    - **The cap bends the interval, never the window.** If the targets exceed `max_frames`, the
      interval is stretched to ``window / (max_frames - 1)`` so the frames still span the whole
      window, and the summary *says so* and names `start`/`end` as the way to look closer. Dropping
      the tail instead would silently answer a different question than the one asked.
    - **Memory stays flat.** The decode is sequential and exactly one decoded frame is held at a
      time, so a long clip costs no more RAM than a short one. The bound on the clip itself is
      `_assets.MAX_VIDEO_BYTES`.
    - **The summary is measured, never assumed.** It reports the timestamps of the frames that were
      actually decoded — which, near the end of a clip, are not the targets that were asked for
      (there is no frame at exactly ``t = duration``). A model reasoning about "frame at 4s" must be
      reading the frame it got, not the one that was requested.

    Raises `ValueError` on undecodable bytes.
    """
    info = probe(video_bytes)
    lo, hi = _window(info, start, end)
    targets, stretched, interval = _targets(lo, hi, every, max_frames)

    frames: list[ImageContent] = []
    stamps: list[float] = []
    with _open(video_bytes) as container:
        stream = _video_stream(container)
        pending = list(targets)
        last: Any = None
        for frame in container.decode(stream):
            if not pending:
                break
            time = frame.time if frame.time is not None else 0.0
            while pending and time >= pending[0] - _EPSILON:
                pending.pop(0)
                _take(frame, time, frames, stamps, name, long_edge)
            last = frame
        # Targets past the last frame (``t = duration`` has no frame of its own) resolve to the
        # last frame there is — the honest answer to "show me the end", and the reason `_take`
        # de-duplicates rather than emitting the same still once per unmet target.
        if pending and last is not None:
            _take(last, last.time or 0.0, frames, stamps, name, long_edge)

    if not frames:
        raise ValueError("the video decoded no frames.")
    return _summary(name, info, frames, stamps, interval, stretched), frames


class WatchVideoTool(PlatformTool):
    """``watch_video`` — put a video asset on the timeline in front of the model.

    A `PlatformTool` read, like the assets tool's `view`: it fetches the asset through the bound
    SDK client and returns it. There is **no provider call and no idempotency key** — nothing is
    created, nothing is spent, nothing is posted — which is why it is a benign default tool rather
    than a powerful opt-in one.

    It returns the clip and says nothing about perception (issue #316). What the model actually
    receives — the video, sampled frames, or an honest caption — is decided by the engine from the
    provider's own declared capabilities.
    """

    name = "watch_video"
    description = (
        "Watch a video file on the timeline. Give the video asset's uuid (find it with the "
        "assets tool's 'list', or say 'latest' for the newest file on the timeline) and the "
        "video is put in front of you to look at — the way 'view' shows you an image and "
        "'listen' reads you audio. This is how you check a video you generated yourself: watch "
        "it and see whether it is what you asked for. Optional 'every' sets the seconds between "
        "the frames you are shown (default 1); 'start' and 'end' (seconds) narrow the window so "
        "you can look closely at one moment of a long clip. A non-video file comes back with a "
        "clean note, not an error."
    )
    parameters = {
        "type": "object",
        "properties": {
            "uuid": {
                "type": "string",
                "description": (
                    "The video asset's uuid, or 'latest' for the newest file on the timeline "
                    "(which is the video you just generated, if you just generated one)."
                ),
            },
            "every": {
                "type": "number",
                "description": (
                    "Optional seconds between sampled frames (default 1). Smaller sees more "
                    "motion; the per-call frame cap still applies, so narrow the window with "
                    "'start'/'end' to actually look closer."
                ),
            },
            "start": {
                "type": "number",
                "description": "Optional start of the window to watch, in seconds from the clip's start.",
            },
            "end": {
                "type": "number",
                "description": "Optional end of the window to watch, in seconds from the clip's start.",
            },
            "timeline": {
                "type": "string",
                "description": "Optional timeline uuid to look on. Defaults to the current timeline.",
            },
        },
        "required": ["uuid"],
    }

    def run(
        self,
        uuid: str | None = None,
        every: float | None = None,
        start: float | None = None,
        end: float | None = None,
        timeline: str | None = None,
    ) -> str | ToolResult:
        """Fetch the video asset and hand it back for the model to actually watch."""
        if not uuid or not uuid.strip():
            return (
                "Error: 'watch_video' needs the video asset's uuid. Use the assets tool's 'list'."
            )
        target = timeline or self.context.timeline
        resolved = resolve_uuid(self.context.client, uuid, target)
        if resolved is None:
            return "No files on this timeline yet — nothing to watch."

        asset = self.context.client.assets.get(resolved)
        meta = _describe(asset)
        file = asset.content.file
        sampling = FrameSampling(
            every=every if every and every > 0 else DEFAULT_EVERY, start=start, end=end
        )
        result = video_input(file, sampling)
        if isinstance(result, str):
            return f"{meta}\n({result})"  # a reason it can't be watched — never raw bytes
        return ToolResult(text=f"{meta}\n({_facts(result)})", videos=[result])


# --- internals ---------------------------------------------------------------

#: Slack when comparing a frame's timestamp against a target. Frame times are rationals converted
#: to float, so an exact-looking ``2.0`` can decode as ``1.9999999999999998`` — without this the
#: frame at a whole second is skipped and the *next* one is taken, which shifts every sampled
#: timestamp by one frame interval and makes the summary quietly wrong.
_EPSILON = 1e-6


def _open(video_bytes: bytes) -> Any:
    """Open the clip's bytes for decoding, mapping any decoder failure to `ValueError`.

    PyAV is imported here rather than at module scope so an import-time failure (a broken wheel,
    a platform with no binary) surfaces as this same readable `ValueError` on the one call that
    needs it, instead of taking down the whole package — and, more to the point, instead of taking
    down every *other* tool with it.
    """
    try:
        import av
    except ImportError as exc:  # pragma: no cover - av is a base dependency
        raise ValueError(f"video decoding is unavailable on this install: {exc}") from exc
    try:
        return av.open(io.BytesIO(video_bytes))
    except Exception as exc:  # every decoder failure is one and the same: an unreadable file
        raise ValueError(f"could not read the video: {exc}") from exc


def _video_stream(container: Any) -> Any:
    """The clip's first video stream, or a `ValueError` naming what is missing."""
    streams = container.streams.video
    if not streams:
        raise ValueError("the file has no video stream.")
    stream = streams[0]
    # Decoding is single-threaded on purpose: a wake is one clip at a time, and a decoder thread
    # pool on a shared fleet box costs more in contention than it saves in wall-clock.
    stream.thread_type = "NONE"
    return stream


def _duration_of(container: Any, stream: Any) -> float:
    """The clip's length in seconds, from the container or the stream, or ``0.0`` if neither says.

    ``0.0`` means *unknown*, never *empty*: the sampler treats it as a window it cannot compute and
    falls back to walking the stream, rather than printing a made-up length.
    """
    import av

    if container.duration:
        return float(container.duration) / av.time_base
    if stream.duration and stream.time_base:
        return float(stream.duration * stream.time_base)
    return 0.0


def _window(info: VideoInfo, start: float | None, end: float | None) -> tuple[float, float]:
    """The clamped ``(start, end)`` of the window to sample, in seconds.

    A window past the end of the clip is clamped rather than refused — an agent guessing at a
    length it was told approximately should get the tail of the clip, not an error. An `end` at or
    before `start` collapses to a single instant, which samples one frame.
    """
    lo = max(0.0, start or 0.0)
    if info.duration_s > 0:
        lo = min(lo, info.duration_s)
        hi = info.duration_s if end is None else min(float(end), info.duration_s)
    else:
        hi = lo if end is None else float(end)
    return lo, max(lo, hi)


def _targets(
    lo: float, hi: float, every: float, max_frames: int
) -> tuple[list[float], bool, float]:
    """The timestamps to sample, whether the cap stretched the interval, and the interval used."""
    span = hi - lo
    max_frames = max(max_frames, 1)
    if span <= 0 or every <= 0 or max_frames == 1:
        return [lo], False, 0.0
    steps = math.floor(span / every + _EPSILON)
    targets = [lo + i * every for i in range(steps + 1)]
    if targets[-1] < hi - _EPSILON:
        targets.append(hi)  # the last frame is always represented
    if len(targets) <= max_frames:
        return targets, False, every
    # Too many: stretch the interval so the frames still span the whole window. Never drop the
    # tail — that would answer a narrower question than the one the agent asked.
    interval = span / (max_frames - 1)
    targets = [lo + i * interval for i in range(max_frames)]
    targets[-1] = hi
    return targets, True, interval


def _stamp(seconds: float) -> str:
    """A frame's timestamp as the model reads it — one decimal where that is exact, else two.

    Two properties, and both were caught by a test rather than reasoned about in advance:

    - **A label must not round to a different moment.** The tail frame of a 5.0s clip is at
      4.958s; ``t=5.0s`` would tell the model it is looking at an instant that has no frame, which
      is the "the summary is measured, never assumed" claim in `sample_frames` broken in the one
      place it matters most (a "does it end right?" question).
    - **Two different frames must not carry the same label.** At an interval finer than a tenth of
      a second, one decimal collapses 0.042s and 0.083s onto ``t=0.0s`` — two distinct stills the
      model is told are the same moment.

    A clean tenth still prints as a clean tenth (``t=0.0s``, ``t=2.0s``), so the ordinary case
    reads exactly as it always did.
    """
    text = f"{seconds:.2f}"
    return text.removesuffix("0")


def _take(
    frame: Any,
    time: float,
    frames: list[ImageContent],
    stamps: list[float],
    name: str,
    long_edge: int,
) -> None:
    """Encode one decoded frame as a JPEG `ImageContent`, skipping a repeat of the one before.

    The de-duplication is not tidiness: several targets can land on the same frame (an `every`
    finer than the frame interval, or two targets past the end of the clip), and emitting the same
    still twice would spend the frame budget on a duplicate while telling the model it was looking
    at two different moments.
    """
    if stamps and abs(stamps[-1] - time) < _EPSILON:
        return
    frames.append(
        ImageContent(url=_jpeg_data_url(frame, long_edge), alt=f"{name} t={_stamp(time)}s")
    )
    stamps.append(time)


def _jpeg_data_url(frame: Any, long_edge: int) -> str:
    """One decoded frame, scaled to the long edge and encoded as a JPEG ``data:`` URL.

    Never scaled **up**: a frame already smaller than `long_edge` is encoded as it is, because
    upscaling spends the model's tokens on pixels the camera never recorded.
    """
    width, height = int(frame.width), int(frame.height)
    longest = max(width, height)
    if longest > long_edge:
        scale = long_edge / longest
        frame = frame.reformat(
            width=max(2, round(width * scale)), height=max(2, round(height * scale))
        )
    buffer = io.BytesIO()
    frame.to_image().save(buffer, format="JPEG", quality=FRAME_QUALITY)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _facts(clip: VideoContent) -> str:
    """The one-line probe summary a `watch_video` result carries: how long, how fast, how big.

    Read off the clip the tool actually fetched, so the model reasons about the real file rather
    than the metadata the platform recorded for it. A clip whose header cannot be parsed says so
    plainly and is still handed on — the engine's tiers can fail more informatively than a tool
    that refused the file outright.
    """
    try:
        info = probe(decode_data_url(clip.url))
    except ValueError as exc:
        return f"video — could not read its header: {exc}"
    parts = [f"{info.duration_s:.1f}s"] if info.duration_s > 0 else ["length unknown"]
    if info.fps:
        parts.append(f"{info.fps:.0f} fps")
    if info.width and info.height:
        parts.append(f"{info.width}x{info.height}")
    return "video — " + ", ".join(parts)


def _summary(
    name: str,
    info: VideoInfo,
    frames: list[ImageContent],
    stamps: list[float],
    interval: float,
    stretched: bool,
) -> str:
    """The caption for a sampled-frames turn: what the clip is, and exactly what was sampled."""
    facts = [f"{info.duration_s:.1f}s"] if info.duration_s > 0 else ["length unknown"]
    if info.fps:
        facts.append(f"{info.fps:.0f} fps")
    if info.width and info.height:
        facts.append(f"{info.width}x{info.height}")
    times = ", ".join(f"t={_stamp(t)}s" for t in stamps)
    plural = "frame" if len(frames) == 1 else "frames"
    every = f" sampled every {interval:.1f}s" if interval > 0 else ""
    line = f"(Showing {len(frames)} {plural} of {name} ({', '.join(facts)}){every}: {times})"
    if stretched:
        line += (
            f" {MAX_FRAMES} frames is the per-call cap, so the interval was stretched to cover "
            "the whole window — narrow it with 'start' and 'end' to look closer."
        )
    return line


def decode_data_url(url: str) -> bytes:
    """The bytes behind a ``data:<type>;base64,<...>`` URL.

    `VideoContent` carries its clip as a data URL for the same reason `ImageContent` does — the
    input is then self-contained and never depends on a vendor reaching a signed blob URL — so the
    frames fallback, which needs the *bytes*, decodes them back out here. Raises `ValueError` on
    anything that is not such a URL, which is the same failure every other decode boundary in this
    module raises.
    """
    marker = ";base64,"
    if not url.startswith("data:") or marker not in url:
        raise ValueError("the video is not an inline data URL.")
    try:
        return base64.b64decode(url.split(marker, 1)[1])
    except Exception as exc:  # a malformed payload is one and the same: an unreadable file
        raise ValueError(f"could not decode the video payload: {exc}") from exc
