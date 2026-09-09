"""Let the agent watch: the sampler, the trim, and the facts behind the assets tool's `watch`.

The video half of the three senses the assets tool carries — `view` an image, `watch` a video,
`listen` to audio (issue #484). Until video perception landed, no harness agent could perceive a
clip at all: `view` is images-only, a posted clip on an `asset.created` wake was acknowledged in
text and never seen, and an agent that generated a video had no way to check its own work. On
2026-08-13 @eddie-murphy generated three clips for @origin and asserted a first-frame match he had
no way to verify — the founder's ruling is that agents get eyes for video, on **every** harness
agent, and that the harness never tells a model to ask a human to look (issue #471).

Three tiers, chosen by capability — never by vendor
----------------------------------------------------
The tool's job ends at *fetching the clip*; how the model perceives it is the **engine's** call,
exactly as it is for an image. The assets tool's `watch` action returns a `VideoContent` and the
engine routes it (`_engine._show_media`):

1. **The model takes video** (`supports_video` answers a definite yes) → it gets the video,
   `native_watch`-trimmed to the window the agent asked for.
2. **The model takes images** → `sample_frames` decodes frames here and the model gets those.
3. **The model takes neither** → an honest caption saying the clip was described, not shown.

This is one seam extended, not a second one built beside it: the same `_split_result` →
`_show_media` → evict path an image already travels. The tool has no view of the provider (the
body/brain split — a `PlatformContext` is client + timeline only), so it deliberately does not
narrate perception (issue #316): claiming "watching it now" would promise sight the tool cannot
verify.

Pure Python, no subprocess
--------------------------
Frames are decoded — and a window is cut — with **PyAV**, whose wheels bundle FFmpeg, so all of it
happens *in this process* and the locked profile's no-shell boundary is untouched. No `ffmpeg`
subprocess, no apt package, nothing for `Policy.locked()` to have an opinion about. That is the
whole reason `watch` rides the benign assets tool rather than the powerful opt-in set: looking at
a file already on the timeline costs no provider call and reaches nothing outside the process.

Frames — and a trimmed clip — exist **in memory only**: never written to disk, never posted as
assets, never replacing the asset on the timeline, and evicted from the transcript after the turn
that showed them, exactly like a viewed image.
"""

from __future__ import annotations

import base64
import io
import logging
import math
from dataclasses import dataclass, replace
from typing import Any

from basecradle_harness._assets import _data_url
from basecradle_harness._messages import FrameSampling, ImageContent, VideoContent

_log = logging.getLogger("basecradle_harness")

#: The most frames one assets `watch` call will put in front of the model. A cap, not a knob: an
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

#: The container and codec a **re-encoded** cut is written into (`cut`). MP4/H.264 is the one
#: pairing every vendor that accepts video takes, and PyAV's wheels bundle libx264 — so the
#: re-encode stays in-process exactly as the frame sampler does, with no subprocess and nothing
#: for the locked profile to have an opinion about.
_CUT_FORMAT = "mp4"
CUT_CONTENT_TYPE = "video/mp4"
_CUT_CODEC = "h264"
_CUT_PIX_FMT = "yuv420p"
#: Constant-quality encode. 23 is x264's own default: visually clean, and a cut is evidence the
#: model looks at once, not a deliverable that goes back on the timeline.
_CUT_CRF = "23"
#: Frame rate to fall back on when a container states none, so a cut of an rate-less stream still
#: has a clock. Only reached for a clip whose header says nothing, which `video_facts` also
#: reports as ``length unknown``.
_CUT_FALLBACK_RATE = 24

#: How near the requested start a keyframe must sit for a **lossless container copy** to be the
#: honest answer to what the agent asked. A copy can only begin at a keyframe, so a copy whose
#: nearest keyframe is further back than this would hand back seconds nobody asked for — at which
#: point the re-encode, which can start on any frame, is the one that answers the question. A
#: quarter of a second is under one sampled-frame interval at the default ``every``.
_KEYFRAME_TOLERANCE = 0.25


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


def asked_span(sampling: FrameSampling) -> str | None:
    """The window the agent asked for, as the model reads it — ``None`` when it asked for none.

    The **single spelling** of a requested window, shared by every clause that names one, so a
    caption saying a window *was* applied and one saying it was not can never word it differently.
    """
    if sampling.start is None and sampling.end is None:
        return None
    if sampling.start is not None and sampling.end is not None:
        return f"{sampling.start:g}s-{sampling.end:g}s"
    if sampling.start is not None:
        return f"from {sampling.start:g}s"
    return f"up to {sampling.end:g}s"


def window_note(sampling: FrameSampling) -> str | None:
    """What to say when a clip goes to the model **whole** though a window was asked for.

    Since issue #482 a window *is* applied on the video-native tier (`native_watch` cuts the clip
    to it), so this is now the **fallback** clause: what a caption says when the cut could not be
    made — an undecodable container, a codec PyAV cannot re-encode, a window that clamps to
    nothing. The frames tier still honors the window itself, so the honest statement is that the
    window narrowed frames and not this clip.

    Before #482 it was the *only* answer, and it is worth remembering why it existed at all: until
    issue #481 nothing said anything, so an agent that asked to look closely at one moment of a
    long clip was shown all of it and had no way to learn which of those two things had happened
    — the #479 shape in a different place, a model reasoning about a perception it did not have.

    ``None`` when no window was asked for, which is the ordinary case: a note about a window nobody
    named is noise on every caption forever.
    """
    span = asked_span(sampling)
    if span is None:
        return None
    return f"the start/end window you asked for ({span}) narrows sampled frames only"


def video_facts(info: VideoInfo) -> str:
    """A clip's header facts as one comma-joined phrase: how long, how fast, how big.

    The **single spelling** behind every line that states them — the assets `watch` result, the
    sampled-frames summary, and the describer's caption for a natively-watched clip (issue #479).
    Three copies of one format string is three chances for a model to read two differently-worded
    accounts of the same file and wonder which is the real one; ``length unknown`` rather than a
    fabricated ``0.0s`` is part of that spelling, because a container that states no duration is a
    legitimately unknowable answer and not a zero-length clip.
    """
    parts = [f"{info.duration_s:.1f}s"] if info.duration_s > 0 else ["length unknown"]
    if info.fps:
        parts.append(f"{info.fps:.0f} fps")
    if info.width and info.height:
        parts.append(f"{info.width}x{info.height}")
    return ", ".join(parts)


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


# --- the native tier's window: cut the clip to what was asked for (issue #482) ---


@dataclass(frozen=True)
class Cut:
    """A clip narrowed to a window: the bytes to send, what they are, and what they cover.

    `start`/`end` are the window the cut **actually** covers, in seconds from the original clip's
    start — not the window that was requested. They can differ (a lossless copy can only begin on
    a keyframe, and a window is clamped to the clip's real length), and the caption states these,
    never the request, for the same reason `sample_frames` reports the timestamps it decoded rather
    than the ones it aimed at: a model reasoning about "second 12" must be reasoning about what it
    was actually shown.

    `lossless` says which of the two paths produced it — a **container copy** (the packets moved
    across untouched; the picture data is the original's, byte for byte) or a **re-encode**. It is
    reported for the log line, never for the model: what the agent asked about is the window, and
    the caption's job is to name that.
    """

    data: bytes
    content_type: str
    start: float
    end: float
    lossless: bool


@dataclass(frozen=True)
class NativeWatch:
    """What the video-native tier actually sends the model, and how the caption should say so.

    Three states, and every caller renders all three (`_engine._video_caption`,
    `_describer._watched_facts`):

    - **no window asked for** — `span` and `clause` are both ``None``; the whole clip goes, and the
      caption is byte-identical to what it was before any of this existed, which is the regression
      bar;
    - **window applied** — `span` names the window actually sent (e.g. ``"3s-5s"``) and `clause` is
      the sentence explaining it; `clip` carries the cut bytes;
    - **window not applied** — `span` is ``None`` and `clause` is `window_note`'s honest fallback;
      `clip` is the original, whole.
    """

    clip: VideoContent
    span: str | None = None
    clause: str | None = None


def native_watch(clip: VideoContent) -> NativeWatch:
    """The clip as the video-native tier should send it, honoring the agent's `start`/`end`.

    The founder's ruling, 2026-09-09 (issue #482, split out of #481): *the harness never modifies
    content, but this is a tool, and if the LLM only wants to watch part of a video the tool should
    let it — it saves money when only part matters, and lets an agent see a video longer than its
    model's maximum by watching it in pieces.* Before it, `start`/`end` narrowed the **sampled-
    frames** tier only: a model that takes video was sent the clip whole, the sampler never ran,
    and an agent on a video-native brain was *less* capable than a vision-only peer making the
    identical call — it paid the whole clip's tokens on every look and got an account of all of it.

    **This is inside the tool's job and outside issue #336's rule, and the line between them is
    who asked.** #336 forbids the harness modifying a file's bytes *to fit a vendor's limit* — no
    downscaling, no recompression, attempt the original honestly and relay the verdict. Here the
    **agent** asked for a window, in the call it made, and cutting to it is the tool doing what it
    was told. Nothing on the timeline changes: the Asset is untouched, the cut lives in memory for
    one turn and is evicted with the rest of the payload.

    Never a silent failure and never a fabrication: a cut that cannot be made (an undecodable
    container, a codec that will not re-encode, a window that clamps to nothing) degrades to the
    whole clip plus `window_note`'s honest clause — exactly the behaviour #481 shipped — with a
    ``WARNING``, because a window the agent asked for and did not get is a perception it must not
    reason about as though it had.
    """
    sampling = clip.sampling
    asked = asked_span(sampling)
    if asked is None:
        return NativeWatch(clip=clip)  # the ordinary case: whole clip, caption unchanged
    try:
        made = cut(decode_data_url(clip.url), sampling.start, sampling.end)
    except ValueError as exc:
        _log.warning(
            "video window not applied to %s (%s): %s — sending the whole clip.",
            clip.alt or "video",
            asked,
            exc,
        )
        return NativeWatch(clip=clip, clause=window_note(sampling))
    span = _span_of(made)
    _log.info(
        "video trimmed to %s of %s (asked %s, %s).",
        span,
        clip.alt or "video",
        asked,
        "container copy" if made.lossless else "re-encoded",
    )
    trimmed = replace(
        clip, url=_data_url(made.content_type, made.data), content_type=made.content_type
    )
    return NativeWatch(clip=trimmed, span=span, clause=_cut_clause(asked, span, made, sampling))


def cut(video_bytes: bytes, start: float | None, end: float | None) -> Cut:
    """Narrow a clip to ``[start, end]`` in memory, losslessly where the container allows.

    Two paths, and the choice between them is about **accuracy, not speed**:

    - **container copy** — the packets in the window are remuxed into a fresh container with their
      timestamps rebased to zero. The picture data is the original's, byte for byte, so nothing is
      recompressed. It is only available from a keyframe, so it is taken only when a keyframe sits
      within `_KEYFRAME_TOLERANCE` of the requested start — which is always true of the common
      ``end``-only request ("the first N seconds"), whose window starts at zero.
    - **re-encode** — the frames in the window are decoded and encoded afresh (H.264 in MP4, via
      the libx264 PyAV's wheels bundle). Costs a generation of quality, and is the only thing that
      can start mid-GOP, which is what "watch seconds 12 to 18 of a long clip" needs.

    **The result is verified before it is returned** (`probe` of the produced bytes). Every claim a
    caption makes about the window rides on these bytes being a real, decodable clip, and a muxer
    that silently produced something a vendor will reject would turn an honest degrade into a hard
    provider failure on a founder-visible path. A verify failure raises, and the caller falls back
    to the whole clip.

    Raises `ValueError` — this module's uniform parse-boundary failure — on undecodable bytes, a
    window that covers nothing, or a cut that will not verify.
    """
    info = probe(video_bytes)
    lo, hi = _window(info, start, end)
    if hi - lo <= _EPSILON:
        raise ValueError(f"the {start!r}-{end!r} window covers none of the clip.")
    keyframe = _keyframe_at_or_before(video_bytes, lo)
    made = None
    if keyframe is not None and lo - keyframe <= _KEYFRAME_TOLERANCE:
        made = _copy_cut(video_bytes, keyframe, hi)
    if made is None:
        made = _encode_cut(video_bytes, lo, hi, info)
    probe(made.data)  # a cut nobody can decode is not a cut — never hand one to a vendor
    return made


# --- internals ---------------------------------------------------------------

#: FFmpeg's fixed container-level time base (``AV_TIME_BASE``, microseconds) — the unit a
#: container-level `seek` offset is expressed in. A constant of the format rather than of a build,
#: so it is spelled once here instead of reached for through an import at three call sites.
_AV_TIME_BASE = 1_000_000

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


def _span_of(made: Cut) -> str:
    """The window a cut actually covers, as the model reads it — hundredths, trailing zeros gone.

    Rounded for the same reason a sampled frame's label is (`_stamp`): a re-encoded window ends on
    the last frame there is, at ``3.9583333333`` seconds, and a caption is a sentence a model reads
    rather than a float it parses. Rounded to a *hundredth*, never a tenth — at a fine interval a
    tenth collapses two different instants onto one number.
    """
    return f"{round(made.start, 2):g}s-{round(made.end, 2):g}s"


def _cut_clause(asked: str, span: str, made: Cut, sampling: FrameSampling) -> str:
    """The caption's sentence for a clip that *was* cut to a window.

    Two wordings, and the second exists because the two windows can honestly differ: a lossless
    copy starts at the nearest keyframe at or before the request, so a caption claiming the exact
    request would be the "reasons about a perception it did not have" defect in miniature. The
    comparison is **numeric** — the strings are formatted differently by construction (``"up to
    5s"`` versus ``"0s-5s"``), so comparing them would report every one-sided request as inexact.
    """
    exact = (
        sampling.start is None or abs(sampling.start - made.start) <= _KEYFRAME_TOLERANCE
    ) and (sampling.end is None or abs(sampling.end - made.end) <= _KEYFRAME_TOLERANCE)
    if exact:
        return f"trimmed to the {span} window you asked for"
    return f"trimmed to {span}, the nearest whole window to the {asked} you asked for"


def _keyframe_at_or_before(video_bytes: bytes, seconds: float) -> float | None:
    """The timestamp of the last keyframe at or before `seconds`, or ``None`` if there is none.

    A container seek lands on exactly that keyframe (``backward=True``), so this is a demux of one
    packet, not a scan. ``None`` for a stream whose packets carry no timestamps — for which a copy
    could not be rebased anyway, so the caller re-encodes.
    """
    try:
        with _open(video_bytes) as source:
            stream = _video_stream(source)
            source.seek(int(max(seconds, 0.0) * _AV_TIME_BASE), backward=True, any_frame=False)
            for packet in source.demux(stream):
                if packet.pts is None or not packet.is_keyframe:
                    continue
                return float(packet.pts * packet.time_base)
    except Exception:  # noqa: BLE001 - an unseekable stream is a re-encode, never a failure
        return None
    return None


def _copy_cut(video_bytes: bytes, lo: float, hi: float) -> Cut | None:
    """Remux the packets in ``[lo, hi]`` into a fresh container — no recompression at all.

    `lo` is a **keyframe** timestamp (the caller found it), which is what makes a copy decodable:
    a stream that begins mid-GOP references frames that are not there. Every video and audio
    stream is carried across, so a clip's sound survives the cut; timestamps are rebased per stream
    so the result starts at zero rather than at `lo`, which is what a player — and a vendor — reads
    as the clip's own clock.

    ``None`` (never an exception) when the copy cannot be built — a codec the output container will
    not take, a muxer that refuses, no packets in the window — so the caller re-encodes instead.
    """
    out = io.BytesIO()
    try:
        with _open(video_bytes) as source:
            streams = [s for s in source.streams if s.type in ("video", "audio")]
            if not any(s.type == "video" for s in streams):
                return None
            source.seek(int(max(lo, 0.0) * _AV_TIME_BASE), backward=True, any_frame=False)
            target = _open_output(out)
            mapping = {s.index: target.add_stream_from_template(s) for s in streams}
            offsets: dict[int, int] = {}
            muxed = 0
            last = lo  # the end reported is what was muxed, never the bound that was asked for
            for packet in source.demux(streams):
                if packet.pts is None or packet.dts is None:
                    continue  # a flush packet, or one with no clock to rebase
                time = float(packet.pts * packet.time_base)
                if time < lo - _EPSILON or time > hi + _EPSILON:
                    if packet.stream.type == "video" and time > hi:
                        break  # past the window on the stream that defines it
                    continue
                index = packet.stream.index
                offsets.setdefault(index, packet.pts)
                packet.pts -= offsets[index]
                packet.dts -= offsets[index]
                if packet.stream.type == "video":
                    last = time
                packet.stream = mapping[index]
                target.mux(packet)
                muxed += 1
            target.close()
    except Exception:  # noqa: BLE001 - any muxer refusal is "copy unavailable", not a failure
        return None
    if not muxed:
        return None
    return Cut(
        data=out.getvalue(), content_type=CUT_CONTENT_TYPE, start=lo, end=last, lossless=True
    )


def _encode_cut(video_bytes: bytes, lo: float, hi: float, info: VideoInfo) -> Cut:
    """Decode the frames in ``[lo, hi]`` and encode them afresh — the any-start path.

    The one thing a container copy cannot do is begin between keyframes, which is exactly what
    "look closely at second 12 of a long clip" asks for. Memory stays flat for the same reason the
    sampler's does: one decoded frame at a time.

    **Audio is not carried.** Re-encoding a second modality would double this path's vendor surface
    for a describer that reads pictures, and a *silent* drop is what would be wrong — so `Cut`
    records `lossless=False` and the log line says which path ran. The copy path, which is the one
    a whole-second-boundary window takes, keeps the audio.

    Presentation timestamps are assigned sequentially at the source's own frame rate, so the cut's
    clock matches the original's. Raises `ValueError` when the window decodes no frames or the
    encoder refuses — the caller degrades to the whole clip.
    """
    from fractions import Fraction

    out = io.BytesIO()
    frames = 0
    first: float | None = None
    last: float = lo
    try:
        with _open(video_bytes) as source:
            stream = _video_stream(source)
            # The stream's own exact rate (a `Fraction`), never `info.fps` — that is a float, and
            # 30000/1001 does not survive the round trip, which would drift a long cut's clock.
            rate = stream.average_rate or stream.guessed_rate or Fraction(_CUT_FALLBACK_RATE)
            source.seek(int(max(lo, 0.0) * _AV_TIME_BASE), backward=True, any_frame=False)
            target = _open_output(out)
            encoder = target.add_stream(_CUT_CODEC, rate=rate)
            encoder.width = int(stream.codec_context.width or info.width)
            encoder.height = int(stream.codec_context.height or info.height)
            encoder.pix_fmt = _CUT_PIX_FMT
            encoder.options = {"crf": _CUT_CRF}
            for frame in source.decode(stream):
                time = frame.time if frame.time is not None else 0.0
                if time < lo - _EPSILON:
                    continue
                if time > hi + _EPSILON:
                    break
                if first is None:
                    first = time
                last = time
                frame.pts = frames
                frame.time_base = 1 / rate
                for packet in encoder.encode(frame):
                    target.mux(packet)
                frames += 1
            for packet in encoder.encode():
                target.mux(packet)
            target.close()
    except Exception as exc:  # every encoder/muxer refusal is one thing: no cut
        raise ValueError(f"could not trim the video: {exc}") from exc
    if not frames:
        raise ValueError("the window decoded no frames.")
    return Cut(
        data=out.getvalue(),
        content_type=CUT_CONTENT_TYPE,
        start=lo if first is None else first,
        end=last,
        lossless=False,
    )


def _open_output(buffer: io.BytesIO) -> Any:
    """An in-memory output container for a cut, in `_CUT_FORMAT`.

    The mirror of `_open`: PyAV is imported at the call rather than at module scope, so an install
    with a broken wheel fails on the one call that needs it instead of taking every other tool down
    with it.
    """
    import av

    return av.open(buffer, mode="w", format=_CUT_FORMAT)


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


def clip_facts(clip: VideoContent) -> str:
    """The one-line probe summary an assets `watch` result carries: how long, how fast, how big.

    Read off the clip the tool actually fetched, so the model reasons about the real file rather
    than the metadata the platform recorded for it. A clip whose header cannot be parsed says so
    plainly and is still handed on — the engine's tiers can fail more informatively than a tool
    that refused the file outright.
    """
    try:
        info = probe(decode_data_url(clip.url))
    except ValueError as exc:
        return f"video — could not read its header: {exc}"
    return "video — " + video_facts(info)


def _summary(
    name: str,
    info: VideoInfo,
    frames: list[ImageContent],
    stamps: list[float],
    interval: float,
    stretched: bool,
) -> str:
    """The caption for a sampled-frames turn: what the clip is, and exactly what was sampled."""
    times = ", ".join(f"t={_stamp(t)}s" for t in stamps)
    plural = "frame" if len(frames) == 1 else "frames"
    every = f" sampled every {interval:.1f}s" if interval > 0 else ""
    line = f"(Showing {len(frames)} {plural} of {name} ({video_facts(info)}){every}: {times})"
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
