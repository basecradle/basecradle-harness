"""Video perception: the sampler, the size/type gates, the window trim, and the assets `watch`
action (issues #471, #482, #484).

No model, no network, and no fixture file on disk: every test **encodes its own clip with PyAV**,
five seconds at 24 fps whose colour changes on each whole second. That makes the two things worth
asserting assertable — a sampled frame's *timestamp* and, by its colour, *which second of the clip
it actually came from* — so these tests pin that the sampler returns the moments it claims to,
rather than merely returning the right number of images.
"""

import base64
import io
import json

import httpx
import pytest
import respx
from basecradle import BaseCradle
from PIL import Image

from basecradle_harness import (
    AssetsTool,
    FrameSampling,
    ToolResult,
    VideoContent,
    cut,
    native_watch,
    probe,
    sample_frames,
)
from basecradle_harness._assets import MAX_VIDEO_BYTES, model_sees_video, video_input
from basecradle_harness._platform import PlatformContext
from basecradle_harness._video import decode_data_url, window_note

BC_URL = "https://api.basecradle.test/v1"
TIMELINE = "019e7754-7d4e-7f50-8162-aaaabbbbcccc"
ASSET_UUID = "019e7754-7d4e-7f50-8162-ddddeeeeffff"

#: One colour per whole second of the fixture clip, so a decoded frame says which second it is from.
SECOND_COLORS = [
    (220, 30, 30),
    (30, 220, 30),
    (30, 30, 220),
    (220, 220, 30),
    (220, 30, 220),
]


def make_video(seconds: int = 5, fps: int = 24, width: int = 160, height: int = 120) -> bytes:
    """An in-memory MP4 whose colour changes on every whole second."""
    import av

    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        for index in range(seconds * fps):
            frame = Image.new("RGB", (width, height), SECOND_COLORS[(index // fps) % 5])
            for packet in stream.encode(av.VideoFrame.from_image(frame)):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return buffer.getvalue()


def second_of(frame) -> int:
    """Which second of the fixture clip a sampled frame came from, read off its colour.

    JPEG is lossy and h264 is lossier, so the match is nearest-colour rather than exact — which is
    the point: the assertion is about *which moment* was sampled, not about pixel fidelity.
    """
    image = Image.open(io.BytesIO(decode_data_url(frame.url))).convert("RGB")
    pixel = image.getpixel((image.width // 2, image.height // 2))
    return min(
        range(len(SECOND_COLORS)),
        key=lambda i: sum((a - b) ** 2 for a, b in zip(pixel, SECOND_COLORS[i])),
    )


# --- probe -------------------------------------------------------------------


def test_probe_reads_duration_fps_and_resolution_off_the_header():
    info = probe(make_video())

    assert info.duration_s == pytest.approx(5.0, abs=0.05)
    assert info.fps == pytest.approx(24.0)
    assert (info.width, info.height) == (160, 120)


def test_probe_on_bytes_that_are_not_a_video_raises_a_readable_error():
    with pytest.raises(ValueError, match="could not read the video"):
        probe(b"this is not a video at all")


def test_probe_on_truncated_video_bytes_raises_rather_than_returning_nonsense():
    with pytest.raises(ValueError):
        probe(make_video()[:200])


# --- sampling ----------------------------------------------------------------


def test_defaults_sample_one_frame_a_second_including_the_first_and_the_last():
    summary, frames = sample_frames(make_video(), name="clip.mp4")

    # A 5s clip at every=1 → t=0,1,2,3,4 and the tail. The tail resolves to the last frame there
    # *is* (4.958s), because no frame exists at exactly t=duration.
    assert len(frames) == 6
    assert [second_of(f) for f in frames] == [0, 1, 2, 3, 4, 4]
    assert [f.alt for f in frames[:5]] == [f"clip.mp4 t={t}.0s" for t in range(5)]
    assert "Showing 6 frames of clip.mp4" in summary
    assert "5.0s, 24 fps, 160x120" in summary
    assert "sampled every 1.0s" in summary
    assert "t=0.0s, t=1.0s, t=2.0s, t=3.0s, t=4.0s, t=4.96s" in summary


def test_a_wider_interval_still_ends_on_the_last_frame():
    _, frames = sample_frames(make_video(), name="clip.mp4", every=2)

    # t=0,2,4 by the interval, then the tail — the end of a clip is always represented, because
    # "does it end the way I asked?" is half of what watching a generated clip is for.
    assert [second_of(f) for f in frames] == [0, 2, 4, 4]


def test_a_window_samples_only_that_window():
    _, frames = sample_frames(make_video(), name="clip.mp4", every=1, start=2, end=4)

    assert [second_of(f) for f in frames] == [2, 3, 4]
    assert [f.alt for f in frames] == ["clip.mp4 t=2.0s", "clip.mp4 t=3.0s", "clip.mp4 t=4.0s"]


def test_a_window_past_the_end_is_clamped_not_refused():
    summary, frames = sample_frames(make_video(), name="clip.mp4", every=1, start=4, end=99)

    # An agent guessing at a length it was told approximately gets the tail, never an error: the
    # window is clamped to 4 → 5.0s, which is the target at t=4 plus the clip's real last frame.
    assert [second_of(f) for f in frames] == [4, 4]
    assert [f.alt for f in frames] == ["clip.mp4 t=4.0s", "clip.mp4 t=4.96s"]
    # 4.96, not 5.0: the label names the frame that exists, never the instant that was asked for.
    assert "t=4.0s, t=4.96s" in summary


def test_the_frame_cap_stretches_the_interval_over_the_whole_window_and_says_so():
    summary, frames = sample_frames(make_video(), name="clip.mp4", every=0.1, max_frames=6)

    # 0.1s over 5s would be 51 frames. The cap bends the *interval* (5/5 = 1.0s), never the
    # window: the frames still span 0 → the end, so the answer covers what was asked.
    assert len(frames) == 6
    assert [second_of(f) for f in frames] == [0, 1, 2, 3, 4, 4]
    assert "the interval was stretched to cover the whole window" in summary
    assert "'start' and 'end'" in summary


def test_an_uncapped_call_does_not_claim_the_interval_was_stretched():
    summary, _ = sample_frames(make_video(), name="clip.mp4", every=1)

    assert "stretched" not in summary


def test_frames_are_downscaled_to_the_long_edge_and_never_upscaled():
    _, big = sample_frames(make_video(width=1920, height=1080), every=2, long_edge=768)
    _, small = sample_frames(make_video(width=160, height=120), every=2, long_edge=768)

    wide = Image.open(io.BytesIO(decode_data_url(big[0].url)))
    assert max(wide.size) == 768
    assert wide.size[0] / wide.size[1] == pytest.approx(1920 / 1080, abs=0.01)
    # Smaller than the long edge is left alone — upscaling would spend tokens on invented pixels.
    assert Image.open(io.BytesIO(decode_data_url(small[0].url))).size == (160, 120)


def test_frames_are_jpeg_data_urls_and_never_touch_the_disk(tmp_path):
    _, frames = sample_frames(make_video(), every=2)

    assert all(f.url.startswith("data:image/jpeg;base64,") for f in frames)
    assert Image.open(io.BytesIO(decode_data_url(frames[0].url))).format == "JPEG"
    assert list(tmp_path.iterdir()) == []  # frames are in memory only — nothing is written


def test_an_interval_finer_than_the_frame_rate_never_shows_the_same_still_twice():
    # every=0.01 over 0.2s asks for 21 targets, but a 24 fps clip holds only 6 frames in that
    # window. Emitting one image per *target* would spend the budget on duplicated stills while
    # labelling them as different moments, so the result is bounded by the frames that exist.
    _, frames = sample_frames(make_video(), name="clip.mp4", every=0.01, start=0, end=0.2)

    assert len(frames) == 6
    # And every label is distinct: one decimal would collapse 0.042s and 0.083s onto "t=0.0s",
    # telling the model two different stills are the same instant.
    assert [f.alt for f in frames] == [
        "clip.mp4 t=0.0s",
        "clip.mp4 t=0.04s",
        "clip.mp4 t=0.08s",
        "clip.mp4 t=0.12s",
        "clip.mp4 t=0.17s",
        "clip.mp4 t=0.21s",
    ]
    assert len({f.alt for f in frames}) == len(frames)


def test_corrupt_bytes_raise_a_readable_error_rather_than_a_decoder_traceback():
    with pytest.raises(ValueError, match="could not read the video"):
        sample_frames(b"\x00\x01\x02 not a video", name="clip.mp4")


def test_decode_data_url_refuses_anything_that_is_not_an_inline_payload():
    with pytest.raises(ValueError, match="not an inline data URL"):
        decode_data_url("https://example.com/clip.mp4")


# --- the asset gates ---------------------------------------------------------


class _File:
    def __init__(self, content_type="video/mp4", byte_size=1000, filename="clip.mp4", url=""):
        self.content_type = content_type
        self.byte_size = byte_size
        self.filename = filename
        self.url = url or f"{BC_URL}/blobs/{ASSET_UUID}"


def test_video_input_refuses_a_non_video_and_points_at_the_right_tool():
    reason = video_input(_File(content_type="image/png"))

    assert isinstance(reason, str)
    assert "not a video" in reason and "'view' for an image" in reason


def test_video_input_refuses_an_unwatchable_container():
    reason = video_input(_File(content_type="video/x-matroska"))

    assert isinstance(reason, str) and "not watchable" in reason


def test_video_input_refuses_an_oversized_clip_without_downloading_it():
    with respx.mock(assert_all_called=False) as mock:
        blob = mock.get(f"{BC_URL}/blobs/{ASSET_UUID}")
        reason = video_input(_File(byte_size=MAX_VIDEO_BYTES + 1))

    assert isinstance(reason, str) and "over the" in reason
    assert not blob.called  # the size gate runs before the fetch, as it does for an image


def test_video_input_refuses_an_empty_file():
    assert "empty file" in video_input(_File(byte_size=0))


def test_video_input_inlines_the_bytes_as_a_data_url_and_carries_the_sampling_request():
    data = make_video(seconds=1)
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/blobs/{ASSET_UUID}").mock(
            return_value=httpx.Response(200, content=data)
        )
        clip = video_input(_File(byte_size=len(data)), FrameSampling(every=2, start=1, end=3))

    assert isinstance(clip, VideoContent)
    assert clip.url.startswith("data:video/mp4;base64,")
    assert decode_data_url(clip.url) == data  # self-contained: no vendor fetches a blob URL
    assert clip.alt == "clip.mp4" and clip.content_type == "video/mp4"
    assert (clip.sampling.every, clip.sampling.start, clip.sampling.end) == (2, 1, 3)


# --- the video capability gate (fail-closed, unlike vision) ------------------


class _Says:
    def __init__(self, answer):
        self._answer = answer

    def supports_video(self):
        if isinstance(self._answer, Exception):
            raise self._answer
        return self._answer


def test_only_a_definite_yes_sends_native_video():
    assert model_sees_video(_Says(True)) is True


@pytest.mark.parametrize(
    "provider",
    [
        _Says(False),
        _Says(None),  # unknown → frames, not a 400
        _Says(RuntimeError("metadata read blew up")),
        object(),  # an adapter that declares no capability at all
    ],
)
def test_everything_short_of_a_definite_yes_falls_back_to_frames(provider):
    # The deliberate opposite of `model_sees_images`' fail-open: a wrong guess here costs a tier
    # that still works, where a video part on a model without video input is a hard 400.
    assert model_sees_video(provider) is False


# --- the tool ----------------------------------------------------------------


JOHN_UUID = "019e7754-7d4e-7f50-8162-111122223333"


def asset_body(uuid=ASSET_UUID, content_type="video/mp4", byte_size=1234, filename="clip.mp4"):
    return {
        "type": "asset",
        "created_at": "2026-09-09T12:00:00.000Z",
        "user": {"uuid": JOHN_UUID, "handle": "john", "name": "John Doe", "kind": "human"},
        "timeline": {"uuid": TIMELINE},
        "content": {
            "uuid": uuid,
            "description": "a short clip",
            "file": {
                "filename": filename,
                "byte_size": byte_size,
                "content_type": content_type,
                "checksum": "Yp9p9C8m6Xv2qS1nKQ0r3w==",
                "url": f"{BC_URL}/blobs/{uuid}",
            },
        },
    }


def asset_response(**kwargs):
    return {"asset": asset_body(**kwargs)}


@pytest.fixture
def tool():
    watcher = AssetsTool()
    client = BaseCradle(token="bc_test_token", base_url=BC_URL)
    watcher.bind(PlatformContext(client=client, timeline=TIMELINE))
    return watcher


def test_watch_video_returns_the_clip_for_the_engine_to_route(tool):
    data = make_video()
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/assets/{ASSET_UUID}").mock(
            return_value=httpx.Response(200, json=asset_response(byte_size=len(data)))
        )
        mock.get(f"{BC_URL}/blobs/{ASSET_UUID}").mock(
            return_value=httpx.Response(200, content=data)
        )
        result = tool.run(action="watch", uuid=ASSET_UUID, every=2, start=1, end=4)

    assert isinstance(result, ToolResult)
    assert len(result.videos) == 1
    assert result.videos[0].alt == "clip.mp4"
    assert (result.videos[0].sampling.every, result.videos[0].sampling.start) == (2, 1)
    # The tool reports the file and the probed facts; it never claims to be watching anything —
    # whether the model actually sees it is the engine's call (issue #316).
    assert "clip.mp4" in result.text
    assert "5.0s, 24 fps, 160x120" in result.text
    assert "watching" not in result.text.lower()


def test_watch_video_resolves_the_latest_alias(tool):
    data = make_video(seconds=1)
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/assets").mock(
            return_value=httpx.Response(200, json={"assets": [asset_body()], "next_cursor": None})
        )
        mock.get(f"{BC_URL}/assets/{ASSET_UUID}").mock(
            return_value=httpx.Response(200, json=asset_response(byte_size=len(data)))
        )
        mock.get(f"{BC_URL}/blobs/{ASSET_UUID}").mock(
            return_value=httpx.Response(200, content=data)
        )
        result = tool.run(action="watch", uuid="latest")

    # The case that needs the alias most: watching the clip the agent itself just generated, whose
    # uuid never reached its context (issue #161).
    assert isinstance(result, ToolResult) and result.videos


def test_watch_video_on_a_non_video_is_a_clean_note_not_a_failure(tool):
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/assets/{ASSET_UUID}").mock(
            return_value=httpx.Response(200, json=asset_response(content_type="image/png"))
        )
        result = tool.run(action="watch", uuid=ASSET_UUID)

    assert isinstance(result, str)
    assert "not a video" in result and "'view' for an image" in result


def test_watch_video_on_an_oversized_clip_says_so_without_fetching_it(tool):
    with respx.mock(assert_all_called=False) as mock:
        mock.get(f"{BC_URL}/assets/{ASSET_UUID}").mock(
            return_value=httpx.Response(200, json=asset_response(byte_size=MAX_VIDEO_BYTES + 1))
        )
        blob = mock.get(f"{BC_URL}/blobs/{ASSET_UUID}")
        result = tool.run(action="watch", uuid=ASSET_UUID)

    assert isinstance(result, str) and "over the" in result
    assert not blob.called


def test_watch_needs_a_uuid(tool):
    assert "'watch' needs the asset's uuid" in tool.run(action="watch")


@pytest.mark.parametrize(
    ("sampling", "span"),
    [
        (FrameSampling(start=10, end=12), "10s-12s"),
        (FrameSampling(start=10), "from 10s"),
        (FrameSampling(end=12), "up to 12s"),
        (FrameSampling(start=1.5, end=2.25), "1.5s-2.25s"),
    ],
)
def test_a_window_the_native_tier_cannot_apply_is_named_rather_than_dropped(sampling, span):
    """Issue #481: the agent asked to look at one moment and was shown all of them.

    `start`/`end` narrow the sampled-frames tier; a model that takes video is sent the clip whole
    and the sampler never runs. Saying nothing leaves the agent reasoning about a perception it did
    not have — the #479 shape in a different place.
    """
    note = window_note(sampling)

    assert note == f"the start/end window you asked for ({span}) narrows sampled frames only"


def test_no_window_asked_for_is_no_note_at_all():
    """The ordinary case, and the regression bar: a note about a window nobody named is noise."""
    assert window_note(FrameSampling()) is None
    assert (
        window_note(FrameSampling(every=0.5)) is None
    )  # `every` is frames-scoped in its own words


def test_watch_offers_the_window_but_never_a_frame_cap():
    # The cap is a constant and the window is the knob (`_video.MAX_FRAMES`): an agent looks closer
    # by narrowing, which is both cheaper and more informative than asking for more frames.
    properties = AssetsTool().parameters["properties"]

    assert {"every", "start", "end"} <= set(properties)
    assert "frames" not in properties and "max_frames" not in properties
    assert json.dumps(AssetsTool().parameters)  # a schema the wire can actually carry


# --- the window trim on the video-native tier (issue #482) --------------------


def _clip(**sampling) -> VideoContent:
    """The fixture clip as a `VideoContent`, with a `FrameSampling` built from `sampling`."""
    data = make_video()
    return VideoContent(
        url=f"data:video/mp4;base64,{base64.b64encode(data).decode('ascii')}",
        alt="clip.mp4",
        content_type="video/mp4",
        sampling=FrameSampling(**sampling),
    )


def _colors_of(data: bytes) -> list[tuple[int, int, int]]:
    """The dominant colour of every decoded frame of a clip — which second each frame came from.

    The same trick the sampler tests use, and the only assertion that actually proves a *trim*: a
    cut that returned the right duration from the wrong seconds would pass every other check.
    """
    import av

    out = []
    with av.open(io.BytesIO(data)) as container:
        for frame in container.decode(container.streams.video[0]):
            out.append(frame.to_image().resize((1, 1)).getpixel((0, 0)))
    return out


def _nearest(color):
    """Which of the fixture's five per-second colours a decoded pixel is closest to."""
    return min(
        range(len(SECOND_COLORS)),
        key=lambda i: sum((a - b) ** 2 for a, b in zip(SECOND_COLORS[i], color)),
    )


def test_cut_returns_the_seconds_that_were_asked_for():
    """The founder's ruling (#482): a window the agent asked for is actually applied.

    Asserted by *colour*, not duration: the fixture's second 2 is blue and second 3 is yellow, so a
    cut of 2s-4s that came back green-and-red would still be 2 seconds long and still be wrong.
    """
    made = cut(decode_data_url(_clip().url), 2.0, 4.0)

    assert 1.8 <= made.end - made.start <= 2.2
    assert probe(made.data).duration_s > 0  # verified, decodable bytes — never a broken container
    seconds = {_nearest(c) for c in _colors_of(made.data)}
    assert seconds <= {2, 3, 4} and 2 in seconds


def test_a_window_that_starts_on_a_keyframe_is_copied_not_re_encoded():
    """The common "watch the first N seconds" shape: lossless, because 0 is always a keyframe."""
    made = cut(decode_data_url(_clip().url), None, 2.0)

    assert made.lossless is True
    assert made.start == 0.0
    assert {_nearest(c) for c in _colors_of(made.data)} <= {0, 1, 2}


def test_a_window_that_starts_mid_gop_is_re_encoded_rather_than_widened():
    """The case a container copy cannot answer — and the reason the re-encode path exists.

    A copy can only begin at a keyframe, so honoring "second 3.4 onwards" by copying would hand
    back seconds the agent did not ask for. Encoding costs a generation of quality and answers the
    actual question; the caption then names the window that was really sent either way.
    """
    made = cut(decode_data_url(_clip().url), 3.4, 4.4)

    assert made.lossless is False
    assert 3.3 <= made.start <= 3.5
    assert {_nearest(c) for c in _colors_of(made.data)} <= {3, 4}


def test_a_window_covering_none_of_the_clip_is_refused_not_guessed():
    with pytest.raises(ValueError):
        cut(decode_data_url(_clip().url), 9.0, 9.5)


def test_native_watch_sends_the_trimmed_clip_and_names_the_window():
    watched = native_watch(_clip(start=2.0, end=4.0))

    assert watched.span == "2s-4s"
    assert watched.clause == "trimmed to the 2s-4s window you asked for"
    assert watched.clip.url != _clip(start=2.0, end=4.0).url  # the cut, not the original
    assert watched.clip.sampling.start == 2.0  # the request rides along, unchanged
    assert probe(decode_data_url(watched.clip.url)).duration_s < 3.0


def test_native_watch_with_no_window_is_byte_identical_to_the_whole_clip():
    """The regression bar: an agent that named no window pays nothing and reads what it always did."""
    clip = _clip()

    watched = native_watch(clip)

    assert watched.clip is clip and watched.span is None and watched.clause is None


def test_a_trim_that_cannot_be_made_degrades_to_the_whole_clip_and_says_so(caplog):
    """Never a silent failure: the #481 clause is the fallback, and it is loud in the log."""
    broken = VideoContent(
        url="data:video/mp4;base64,AAAAAA==", alt="bad.mp4", sampling=FrameSampling(start=1.0)
    )

    with caplog.at_level("WARNING", logger="basecradle_harness"):
        watched = native_watch(broken)

    assert watched.clip is broken and watched.span is None
    assert watched.clause == window_note(broken.sampling)
    assert "window not applied" in caplog.text


def test_a_keyframe_snapped_window_says_it_is_the_nearest_one_not_the_one_asked_for():
    """A copy begins at a keyframe, so the clause must not claim the exact request (#479's shape).

    Driven through `cut` + `_cut_clause` rather than a real snapped clip, because whether a given
    encoder puts a keyframe at a given instant is the encoder's business, not this contract's.
    """
    from basecradle_harness._video import Cut, _cut_clause

    made = Cut(data=b"", content_type="video/mp4", start=2.0, end=4.0, lossless=True)

    exact = _cut_clause("2s-4s", "2s-4s", made, FrameSampling(start=2.0, end=4.0))
    snapped = _cut_clause("3s-4s", "2s-4s", made, FrameSampling(start=3.0, end=4.0))

    assert exact == "trimmed to the 2s-4s window you asked for"
    assert snapped == "trimmed to 2s-4s, the nearest whole window to the 3s-4s you asked for"


def test_the_span_a_caption_names_is_rounded_not_a_raw_float():
    """A caption is a sentence a model reads, not a float it parses.

    A re-encoded window ends on the last frame there is — 3.9583333333s — and naming that verbatim
    is noise. Rounded to a **hundredth**, never a tenth: at a fine interval a tenth collapses two
    different instants onto one number, which is the `_stamp` lesson in a second place.
    """
    from basecradle_harness._video import Cut, _span_of

    made = Cut(data=b"", content_type="video/mp4", start=2.0, end=3.9583333333, lossless=False)

    assert _span_of(made) == "2s-3.96s"


def test_a_copied_window_reports_the_end_it_muxed_not_the_bound_it_was_given():
    """The `sample_frames` discipline applied to the trim: report what happened, not what was asked.

    A window is clamped to the clip, and packets land where they land — so the caption's end comes
    off the last video packet actually written, never off the upper bound handed in.
    """
    made = cut(decode_data_url(_clip().url), None, 2.0)

    assert made.end <= 2.0 + 1e-6
    assert made.end > 1.5  # ...and it really did cover the window, not stop early


def test_an_encoded_cut_keeps_the_sources_frame_rate():
    """The clock has to match, or a long cut drifts against the timestamps the agent was given."""
    source = probe(decode_data_url(_clip().url))

    made = cut(decode_data_url(_clip().url), 3.4, 4.4)

    assert abs(probe(made.data).fps - source.fps) < 0.01
