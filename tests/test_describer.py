"""The blind-model describer: a second model's eyes, for a brain that has none (issue #472).

No network and no real model: a fake describer provider records the turn it was handed and returns
a canned description, so these tests pin the two things that matter — **what the describer is
shown** (media plus the fixed harness prompt, no tools) and **what the brain is then told** (a
caption that always names the describer, never a claim that the brain saw pixels).

The regression bar is the first test in the file: with `HARNESS_DESCRIBER_MODEL` unset, every path
is byte-identical to what it was before this feature existed.
"""

import base64
import logging

import pytest

from basecradle_harness import (
    Describer,
    Engine,
    FrameSampling,
    ImageContent,
    Message,
    Tool,
    ToolCall,
    ToolRegistry,
    ToolResult,
    VideoContent,
    describer_from_env,
)
from basecradle_harness._describer import (
    DESCRIBE_FRAMES_SUFFIX,
    DESCRIBE_PROMPT,
    DESCRIBE_VIDEO_SUFFIX,
    DESCRIBER_API_KEY_VAR,
    DESCRIBER_MODEL_VAR,
    DESCRIBER_PROVIDERS_VAR,
    described_caption,
    describer_providers_from_env,
)

DESCRIPTION = "A tabby cat asleep on a windowsill, with the word HELLO written on the glass."


class FakeDescriberProvider:
    """A vision-capable model that answers with a fixed description and records what it was sent."""

    provider = "openrouter"
    model = "google/gemini-3-flash"

    def __init__(self, *, answer=DESCRIPTION, video=False, raises=None):
        self._answer = answer
        self._video = video
        self._raises = raises
        self.seen = []

    def supports_vision(self):
        return True

    def supports_video(self):
        return self._video

    def chat(self, messages, tools=None):
        self.seen.append((messages, tools))
        if self._raises is not None:
            raise self._raises
        return Message.assistant(content=self._answer)


class BlindProvider:
    """The brain: a model that definitely has no image input (the @glm-5.2 case)."""

    provider = "openrouter"
    model = "z-ai/glm-5.2"

    def __init__(self, *replies):
        self._replies = list(replies)
        self.seen = []

    def supports_vision(self):
        return False

    def chat(self, messages, tools=None):
        self.seen.append([_snapshot(m) for m in messages])
        return self._replies.pop(0)


def _snapshot(message):
    return Message(
        role=message.role,
        content=message.content,
        tool_calls=list(message.tool_calls),
        tool_call_id=message.tool_call_id,
        images=list(message.images),
        videos=list(message.videos),
        injected=message.injected,
    )


class ViewTool(Tool):
    name = "view"
    description = "Look at an image."

    def run(self, **kwargs):
        return ToolResult(
            text="cat.png (image/png, 2048 bytes)",
            images=[ImageContent(url="data:image/png;base64,AAAA", alt="cat.png")],
        )


def _clip():
    from tests.test_video import make_video

    return VideoContent(
        url="data:video/mp4;base64," + base64.b64encode(make_video(seconds=3)).decode(),
        alt="clip.mp4",
        content_type="video/mp4",
        sampling=FrameSampling(),
    )


class WatchTool(Tool):
    name = "watch_video"
    description = "Watch a video."

    def run(self, **kwargs):
        return ToolResult(text="clip.mp4 — 3.0s, 24 fps, 160x120", videos=[_clip()])


def _engine(brain, tool, describer=None):
    registry = ToolRegistry()
    registry.register(tool)
    engine = Engine(brain, registry)
    # The engine memoizes its describer lazily; seeding it is how a test supplies a fake without
    # standing up the env-driven adapter factory (which is exercised separately, below).
    engine._describer = describer
    return engine


def _turn(tool_name):
    return (
        Message.assistant(tool_calls=[ToolCall(id="c1", name=tool_name, arguments={})]),
        Message.assistant(content="Then I know what it is."),
    )


# --- off by absence: the regression bar --------------------------------------


def test_with_no_describer_configured_a_blind_model_gets_exactly_the_old_caption():
    """The shipped default. Byte-identical to the pre-#472 behavior — this is the regression bar."""
    brain = BlindProvider(*_turn("view"))
    engine = _engine(brain, ViewTool(), describer=None)
    history = [Message.user("look at cat.png")]

    engine.run(history)

    note = next(m for m in history if m.injected)
    assert note.content == (
        "(No image input on this model — cat.png was described above, not shown.)"
    )
    assert not any(m.images for m in history)


def test_the_model_id_is_the_only_switch(monkeypatch):
    for value in ("", "   "):
        monkeypatch.setenv(DESCRIBER_MODEL_VAR, value)
        assert describer_from_env() is None  # whitespace is absence, not a model id
    monkeypatch.delenv(DESCRIBER_MODEL_VAR, raising=False)
    assert describer_from_env() is None


# --- the config trio: a model without its key or its routing is DEAD, not OFF -----------------


def _env(**overrides):
    base = {
        DESCRIBER_MODEL_VAR: "google/gemini-3-flash",
        DESCRIBER_API_KEY_VAR: "sk-or-v1-describer",
        DESCRIBER_PROVIDERS_VAR: "google-vertex, deepinfra",
    }
    base.update(overrides)
    return {k: v for k, v in base.items() if v is not None}


@pytest.mark.parametrize(
    ("missing", "reason"),
    [
        (DESCRIBER_API_KEY_VAR, "config:missing_api_key"),
        (DESCRIBER_PROVIDERS_VAR, "config:missing_providers"),
    ],
)
def test_a_model_without_its_key_or_routing_is_a_faulted_describer_not_none(missing, reason):
    """Nobody configures a describer by accident — so configured-and-dead is a defect to page on.

    Returning ``None`` here would make a half-configured describer indistinguishable from a
    deliberately blind agent: the exact Green-While-Absent shape this repo names.
    """
    describer = describer_from_env(_env(**{missing: None}))

    assert describer is not None
    assert describer.fault == reason
    assert describer.provider is None  # it never calls anything


def test_a_faulted_describer_reports_at_error_and_answers_nothing(caplog):
    describer = describer_from_env(_env(**{DESCRIBER_API_KEY_VAR: None}))

    with caplog.at_level(logging.DEBUG, logger="basecradle_harness"):
        assert (
            describer.describe_images([ImageContent(url="data:image/png;base64,A", alt="a.png")])
            is None
        )

    records = [r for r in caplog.records if "describer failed" in r.getMessage()]
    assert [r.levelno for r in records] == [logging.ERROR]  # config-class pages
    assert "reason=config:missing_api_key" in records[0].getMessage()


def test_a_config_fault_is_reported_once_per_wake_and_then_drops_to_debug(caplog):
    """The object's life *is* the wake, so "once per wake" needs no clock — and one defect must
    not become a storm on an agent that views ten pictures."""
    describer = describer_from_env(_env(**{DESCRIBER_PROVIDERS_VAR: None}))
    image = [ImageContent(url="data:image/png;base64,A", alt="a.png")]

    with caplog.at_level(logging.DEBUG, logger="basecradle_harness"):
        for _ in range(3):
            describer.describe_images(image)

    levels = [r.levelno for r in caplog.records if "describer failed" in r.getMessage()]
    assert levels == [logging.ERROR, logging.DEBUG, logging.DEBUG]


def test_a_provider_that_will_not_build_is_a_config_fault_carrying_the_reason(monkeypatch, caplog):
    # `AI_SDK` is the *brain's* axis, so it is read from the process environment and never from the
    # mapping — the describer is defined as the brain's stack with a different model, key and pin.
    monkeypatch.setenv("AI_SDK", "a-sdk-that-does-not-exist")
    describer = describer_from_env(_env())

    assert describer.fault == "config:no_provider"
    with caplog.at_level(logging.ERROR, logger="basecradle_harness"):
        assert describer.describe_images([ImageContent(url="x", alt="a.png")]) is None
    line = next(r.getMessage() for r in caplog.records if "describer failed" in r.getMessage())
    assert "reason=config:no_provider" in line
    assert "a-sdk-that-does-not-exist" in line  # the vendor's/adapter's own words, relayed


def test_building_a_describer_logs_nothing(caplog):
    """`describer_from_env` is side-effect-free: every fault reports at the point of *use*, so a
    resolution-only read (`--resolved-config`) never writes a line."""
    with caplog.at_level(logging.DEBUG, logger="basecradle_harness"):
        describer_from_env(_env(**{DESCRIBER_API_KEY_VAR: None}))
        describer_from_env(_env(**{DESCRIBER_PROVIDERS_VAR: None}))

    assert not [r for r in caplog.records if "describer" in r.getMessage()]


def test_the_provider_list_is_parsed_like_the_reranks(monkeypatch):
    monkeypatch.setenv(DESCRIBER_PROVIDERS_VAR, " google-vertex , ,DeepInfra,")
    # Order preserved (OpenRouter reads `only` as a list), blanks dropped, case left as written —
    # the value is sent to OpenRouter, not compared here.
    assert describer_providers_from_env() == ("google-vertex", "DeepInfra")
    monkeypatch.delenv(DESCRIBER_PROVIDERS_VAR)
    assert describer_providers_from_env() == ()


# --- images ------------------------------------------------------------------


def test_a_blind_model_is_handed_the_describers_words():
    brain = BlindProvider(*_turn("view"))
    vision = FakeDescriberProvider()
    engine = _engine(brain, ViewTool(), Describer(vision, "google/gemini-3-flash"))
    history = [Message.user("look at cat.png")]

    engine.run(history)

    note = next(m for m in history if m.injected)
    assert note.content == (
        "(This model has no image input. cat.png was described by google/gemini-3-flash:)\n"
        f"{DESCRIPTION}"
    )
    # The caption names the describer, always: the brain must never be able to read this back as
    # its own perception, and neither must anyone reading its memory later.
    assert "google/gemini-3-flash" in note.content
    assert note.images == [] and note.videos == []  # text only — nothing to evict


def test_the_describer_is_shown_the_pixels_and_the_fixed_prompt_and_no_tools():
    brain = BlindProvider(*_turn("view"))
    vision = FakeDescriberProvider()
    engine = _engine(brain, ViewTool(), Describer(vision, "google/gemini-3-flash"))

    engine.run([Message.user("look")])

    (messages, tools) = vision.seen[0]
    assert len(messages) == 1 and messages[0].role == "user"
    assert messages[0].content == DESCRIBE_PROMPT
    assert messages[0].images[0].url == "data:image/png;base64,AAAA"
    # No tools: a describer with tools is an agent, and this is a sense organ.
    assert tools is None


def test_the_brain_never_receives_the_pixels_even_with_a_describer():
    """The describer sees them; the brain does not. A video/image part on a blind model is a 400."""
    brain = BlindProvider(*_turn("view"))
    engine = _engine(brain, ViewTool(), Describer(FakeDescriberProvider(), "d/model"))

    engine.run([Message.user("look")])

    assert not any(m.images or m.videos for m in brain.seen[1])


def test_a_describer_that_raises_falls_back_to_the_withheld_caption(caplog):
    """Never a fabricated description: the agent is returned to the state it was already in."""
    brain = BlindProvider(*_turn("view"))
    vision = FakeDescriberProvider(raises=RuntimeError("upstream 503"))
    engine = _engine(brain, ViewTool(), Describer(vision, "d/model"))
    history = [Message.user("look")]

    with caplog.at_level(logging.WARNING, logger="basecradle_harness"):
        engine.run(history)

    note = next(m for m in history if m.injected)
    assert note.content == (
        "(No image input on this model — cat.png was described above, not shown.)"
    )
    line = next(r.getMessage() for r in caplog.records if "describer failed" in r.getMessage())
    # Runtime-class: WARNING, not ERROR — this can succeed next time unchanged.
    assert "reason=provider_error" in line and "upstream 503" in line and "d/model" in line


def test_an_empty_description_is_a_failure_not_a_blank_caption(caplog):
    brain = BlindProvider(*_turn("view"))
    engine = _engine(brain, ViewTool(), Describer(FakeDescriberProvider(answer="   "), "d/model"))
    history = [Message.user("look")]

    with caplog.at_level(logging.WARNING, logger="basecradle_harness"):
        engine.run(history)

    assert "described above, not shown" in next(m for m in history if m.injected).content
    assert any("reason=empty_response" in r.getMessage() for r in caplog.records)


def test_a_working_describer_logs_at_info_not_warning(caplog):
    """The WARNING belongs to the degrade this replaced — a healthy agent must not look broken."""
    brain = BlindProvider(*_turn("view"))
    engine = _engine(brain, ViewTool(), Describer(FakeDescriberProvider(), "d/model"))

    with caplog.at_level(logging.DEBUG, logger="basecradle_harness"):
        engine.run([Message.user("look")])

    described = [r for r in caplog.records if "media described" in r.getMessage()]
    assert [r.levelno for r in described] == [logging.INFO]
    assert "describer=d/model" in described[0].getMessage()
    assert not [r for r in caplog.records if "withheld" in r.getMessage()]


# --- video -------------------------------------------------------------------


def test_a_video_describer_watches_the_clip_itself():
    brain = BlindProvider(*_turn("watch_video"))
    vision = FakeDescriberProvider(video=True)
    engine = _engine(brain, WatchTool(), Describer(vision, "d/video-model"))
    history = [Message.user("watch it")]

    engine.run(history)

    (messages, _) = vision.seen[0]
    assert messages[0].videos[0].alt == "clip.mp4"
    assert not messages[0].images  # natively watched — never sampled
    assert messages[0].content == DESCRIBE_PROMPT + DESCRIBE_VIDEO_SUFFIX
    assert "watched by d/video-model" in next(m for m in history if m.injected).content


# --- issue #479: a blind brain gets the structure a sighted one gets ---------


@pytest.mark.parametrize("video", [True, False])
def test_a_video_describer_is_asked_for_first_frame_over_time_and_last_frame(video):
    """The fix in one assertion, on **both** video paths.

    The live run's defect was shape, not capability: one composite paragraph ("the entire image
    vibrates") with no first frame, no last frame and no clock, so the brain could not answer
    "what is in frame 0, and what changed?" — a question the frames path answers for a *sighted*
    brain by construction. Native and sampled must ask for the same three parts, or a blind agent's
    answers depend on which tier its describer happened to land on.
    """
    brain = BlindProvider(*_turn("watch_video"))
    vision = FakeDescriberProvider(video=video)
    engine = _engine(brain, WatchTool(), Describer(vision, "d/model"))

    engine.run([Message.user("watch it")])

    prompt = vision.seen[0][0][0].content
    assert "'First frame:'" in prompt
    assert "'Over time:'" in prompt
    assert "'Last frame:'" in prompt
    assert "timestamps in seconds" in prompt


def test_a_still_is_not_asked_for_the_video_structure():
    """The other half of the pin: a photograph has no first frame, no last frame and no clock.

    Without this, "add the labels to the prompt" quietly becomes "add them to *every* prompt", and
    a describer asked for a clip's timeline over one still answers a question nobody asked.
    """
    brain = BlindProvider(*_turn("view"))
    vision = FakeDescriberProvider()
    engine = _engine(brain, ViewTool(), Describer(vision, "d/model"))

    engine.run([Message.user("look")])

    prompt = vision.seen[0][0][0].content
    assert prompt == DESCRIBE_PROMPT
    for label in ("First frame:", "Over time:", "Last frame:"):
        assert label not in prompt


def test_a_natively_watched_clip_carries_its_own_facts_ahead_of_the_description():
    """Duration, frame rate and resolution — the other half of what the brain could not state.

    The frames path has carried them since #471 (`sample_frames` returns the summary); the native
    path decoded nothing and so said nothing, and a brain asked "how long is it?" about a clip it
    had just been described could only guess.
    """
    brain = BlindProvider(*_turn("watch_video"))
    vision = FakeDescriberProvider(video=True)
    engine = _engine(brain, WatchTool(), Describer(vision, "d/video-model"))
    history = [Message.user("watch it")]

    engine.run(history)

    note = next(m for m in history if m.injected).content
    assert "(Watched the whole of clip.mp4 (3.0s, 24 fps, 160x120).)" in note
    # Ahead of the description, so the brain reads what the clip *is* before what it shows.
    assert note.index("Watched the whole of") < note.index(DESCRIPTION)


def test_a_described_clip_names_a_window_the_native_tier_could_not_apply():
    """Issue #481, on the describer's own native path — the same clause, from the same helper.

    A blind brain has even less to go on than a sighted one: it cannot look and see it got the
    whole clip.
    """
    clip = _clip()
    clip.sampling = FrameSampling(start=1, end=2)
    describer = Describer(FakeDescriberProvider(video=True), "d/video-model")

    described = describer.describe_video(clip)

    assert described.startswith(
        "(Watched the whole of clip.mp4 (3.0s, 24 fps, 160x120) — the start/end window you asked "
        "for (1s-2s) narrows sampled frames only.)"
    )


def test_an_unprobeable_clip_costs_the_facts_line_and_never_the_description():
    """A header that will not parse is a fact about this decoder, not about the clip.

    The describer has already watched it and answered; dropping that answer over a probe — or
    replacing it with a note about the parse — would spend the valuable half to report the cheap
    one.
    """
    clip = VideoContent(url="data:video/mp4;base64,bm90YXZpZGVv", alt="broken.mp4")
    describer = Describer(FakeDescriberProvider(video=True), "d/video-model")

    assert describer.describe_video(clip) == DESCRIPTION


def test_a_vision_only_describer_reads_sampled_frames_and_is_told_they_are_a_sequence():
    brain = BlindProvider(*_turn("watch_video"))
    vision = FakeDescriberProvider(video=False)
    engine = _engine(brain, WatchTool(), Describer(vision, "d/vision-model"))
    history = [Message.user("watch it")]

    engine.run(history)

    (messages, _) = vision.seen[0]
    assert not messages[0].videos
    assert [i.alt for i in messages[0].images] == [
        "clip.mp4 t=0.0s",
        "clip.mp4 t=1.0s",
        "clip.mp4 t=2.0s",
        "clip.mp4 t=2.96s",
    ]
    # Told they are one clip's moments, not four unrelated pictures.
    assert messages[0].content == DESCRIBE_PROMPT + DESCRIBE_FRAMES_SUFFIX
    # The frames summary rides along, so the brain reads what was sampled as well as what was seen.
    note = next(m for m in history if m.injected)
    assert "Showing 4 frames of clip.mp4" in note.content
    assert DESCRIPTION in note.content


def test_the_describers_video_gate_is_the_same_fail_closed_one_the_brain_uses():
    """A describer that cannot answer the capability question reads frames, never a 400."""

    class Unknown(FakeDescriberProvider):
        def supports_video(self):
            return None

    brain = BlindProvider(*_turn("watch_video"))
    vision = Unknown()
    engine = _engine(brain, WatchTool(), Describer(vision, "d/model"))

    engine.run([Message.user("watch it")])

    assert vision.seen[0][0][0].images and not vision.seen[0][0][0].videos


# --- the caption -------------------------------------------------------------


def test_the_caption_names_the_describer_and_says_the_brain_could_not_see():
    caption = described_caption("cat.png", "d/model", "a cat")

    assert caption == "(This model has no image input. cat.png was described by d/model:)\na cat"


@pytest.mark.parametrize("subject", ["cat.png", "clip.mp4", "a.png, b.png"])
def test_the_caption_carries_whatever_subject_it_is_given(subject):
    assert f"{subject} was described by" in described_caption(subject, "d/model", "x")


def test_the_video_caption_says_it_holds_an_account_of_the_whole_clip_not_the_brains_sight():
    """A still is a moment; a clip is a span, and the brain holds one model's account of all of it.

    Structural, not left to instinct: an agent that relays those sentences as its own sight is the
    failure this wording forecloses (issue #479).
    """
    caption = described_caption("clip.mp4", "d/model", "First frame: a cat.", video=True)

    assert caption == (
        "(This model has no video input. clip.mp4 was watched by d/model, and what follows is "
        "that model's description of the clip as a whole — its account of it, not your own "
        "sight:)\nFirst frame: a cat."
    )


def test_a_vendor_error_carrying_the_key_is_redacted_before_it_is_logged(caplog):
    """`detail` relays the vendor's own words, and a vendor may echo the credential back.

    `kv` scrubs, bounds and quotes every value, so this holds by construction rather than by a
    filter here — which is exactly why it is pinned: the guarantee lives in a module this one does
    not own, and a change there would break it silently.
    """
    secret = "sk-or-v1-0123456789abcdef0123456789abcdef"

    class _Leaky(FakeDescriberProvider):
        def chat(self, messages, tools=None):
            raise RuntimeError(f"401 from upstream for key {secret}")

    describer = Describer(_Leaky(), "d/model")
    with caplog.at_level(logging.WARNING, logger="basecradle_harness"):
        assert describer.describe_images([ImageContent(url="x", alt="a.png")]) is None

    line = next(r.getMessage() for r in caplog.records if "describer failed" in r.getMessage())
    assert secret not in line
    assert "[redacted]" in line
