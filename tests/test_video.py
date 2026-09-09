"""Video perception: the sampler, the size/type gates, and the `watch_video` tool (issue #471).

No model, no network, and no fixture file on disk: every test **encodes its own clip with PyAV**,
five seconds at 24 fps whose colour changes on each whole second. That makes the two things worth
asserting assertable — a sampled frame's *timestamp* and, by its colour, *which second of the clip
it actually came from* — so these tests pin that the sampler returns the moments it claims to,
rather than merely returning the right number of images.
"""

import io
import json

import httpx
import pytest
import respx
from basecradle import BaseCradle
from PIL import Image

from basecradle_harness import (
    FrameSampling,
    ToolResult,
    VideoContent,
    WatchVideoTool,
    probe,
    sample_frames,
)
from basecradle_harness._assets import MAX_VIDEO_BYTES, model_sees_video, video_input
from basecradle_harness._platform import PlatformContext
from basecradle_harness._video import decode_data_url

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
    assert "not a video" in reason and "'view' for images" in reason


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
    watcher = WatchVideoTool()
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
        result = tool.run(uuid=ASSET_UUID, every=2, start=1, end=4)

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
        result = tool.run(uuid="latest")

    # The case that needs the alias most: watching the clip the agent itself just generated, whose
    # uuid never reached its context (issue #161).
    assert isinstance(result, ToolResult) and result.videos


def test_watch_video_on_a_non_video_is_a_clean_note_not_a_failure(tool):
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BC_URL}/assets/{ASSET_UUID}").mock(
            return_value=httpx.Response(200, json=asset_response(content_type="image/png"))
        )
        result = tool.run(uuid=ASSET_UUID)

    assert isinstance(result, str)
    assert "not a video" in result and "'view' for images" in result


def test_watch_video_on_an_oversized_clip_says_so_without_fetching_it(tool):
    with respx.mock(assert_all_called=False) as mock:
        mock.get(f"{BC_URL}/assets/{ASSET_UUID}").mock(
            return_value=httpx.Response(200, json=asset_response(byte_size=MAX_VIDEO_BYTES + 1))
        )
        blob = mock.get(f"{BC_URL}/blobs/{ASSET_UUID}")
        result = tool.run(uuid=ASSET_UUID)

    assert isinstance(result, str) and "over the" in result
    assert not blob.called


def test_watch_video_needs_a_uuid(tool):
    assert "needs the video asset's uuid" in tool.run()


def test_watch_videos_parameters_offer_the_window_but_never_a_frame_cap():
    # The cap is a constant and the window is the knob (`_video.MAX_FRAMES`): an agent looks closer
    # by narrowing, which is both cheaper and more informative than asking for more frames.
    properties = WatchVideoTool.parameters["properties"]

    assert set(properties) == {"uuid", "every", "start", "end", "timeline"}
    assert WatchVideoTool.parameters["required"] == ["uuid"]
    assert json.dumps(WatchVideoTool.parameters)  # a schema the wire can actually carry
