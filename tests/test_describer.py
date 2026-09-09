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
    DESCRIBER_MODEL_VAR,
    described_caption,
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
    monkeypatch.delenv(DESCRIBER_MODEL_VAR, raising=False)
    assert describer_from_env() is None
    monkeypatch.setenv(DESCRIBER_MODEL_VAR, "")
    assert describer_from_env() is None
    monkeypatch.setenv(DESCRIBER_MODEL_VAR, "   ")
    assert describer_from_env() is None  # whitespace is absence, not a model id


def test_a_configured_but_unbuildable_describer_is_an_error_and_never_a_raise(monkeypatch, caplog):
    """Config-class failure: dead until a human acts, so ERROR — and the wake still runs."""
    monkeypatch.setenv(DESCRIBER_MODEL_VAR, "some/model")
    monkeypatch.setenv("AI_SDK", "a-sdk-that-does-not-exist")

    with caplog.at_level(logging.ERROR, logger="basecradle_harness"):
        assert describer_from_env() is None

    line = next(r.getMessage() for r in caplog.records if "describer unavailable" in r.getMessage())
    assert "some/model" in line


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
    assert "upstream 503" in line and "d/model" in line


def test_an_empty_description_is_a_failure_not_a_blank_caption(caplog):
    brain = BlindProvider(*_turn("view"))
    engine = _engine(brain, ViewTool(), Describer(FakeDescriberProvider(answer="   "), "d/model"))
    history = [Message.user("look")]

    with caplog.at_level(logging.WARNING, logger="basecradle_harness"):
        engine.run(history)

    assert "described above, not shown" in next(m for m in history if m.injected).content
    assert any("returned no text" in r.getMessage() for r in caplog.records)


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
    assert messages[0].content == DESCRIBE_PROMPT
    assert "described by d/video-model" in next(m for m in history if m.injected).content


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
