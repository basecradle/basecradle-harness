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
    IMAGE_KIND,
    IMAGE_OUTPUT_BUDGET,
    RETRY_BUDGET_FACTOR,
    VIDEO_KIND,
    VIDEO_OUTPUT_BUDGET,
    VIDEO_PART_LABELS,
    described_caption,
    describer_providers_from_env,
)

DESCRIPTION = "A tabby cat asleep on a windowsill, with the word HELLO written on the glass."

#: What a describer that **followed the instruction** answers about a clip: the three labelled
#: parts issue #479 asks for and issue #488 now checks for. A one-paragraph answer is no longer a
#: usable video description, so a fake that gave one would fail every video test for the wrong
#: reason — it would be pinning the check, not the plumbing the test is about.
VIDEO_DESCRIPTION = (
    "First frame: a tabby cat asleep on a windowsill, HELLO on the glass.\n"
    "Over time: at about 1s the cat lifts its head; by 2s it has stood up.\n"
    "Last frame: the windowsill is empty and the glass is fogged."
)


class FakeDescriberProvider:
    """A vision-capable model that answers with a fixed description and records what it was sent."""

    provider = "openrouter"
    model = "google/gemini-3-flash"

    def __init__(self, *, answer=None, video=False, raises=None):
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
        return Message.assistant(content=self._answer_for(messages))

    def _answer_for(self, messages):
        """An explicit `answer`, else one that answers the question actually asked.

        The medium is read off the **prompt**, not off `supports_video`: the sampled-frames tier
        asks a vision-only describer for the same three parts, so keying on the capability would
        hand that path a still's answer and fail it on a check it should pass.
        """
        if self._answer is not None:
            return self._answer
        asked_for_parts = VIDEO_PART_LABELS[0] in (messages[0].content or "")
        return VIDEO_DESCRIPTION if asked_for_parts else DESCRIPTION


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

    records = [
        r
        for r in caplog.records
        if "purpose=helper" in r.getMessage() and "outcome=fallback" in r.getMessage()
    ]
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

    levels = [
        r.levelno
        for r in caplog.records
        if "purpose=helper" in r.getMessage() and "outcome=fallback" in r.getMessage()
    ]
    assert levels == [logging.ERROR, logging.DEBUG, logging.DEBUG]


def test_a_provider_that_will_not_build_is_a_config_fault_carrying_the_reason(monkeypatch, caplog):
    # `AI_SDK` is the *brain's* axis, so it is read from the process environment and never from the
    # mapping — the describer is defined as the brain's stack with a different model, key and pin.
    monkeypatch.setenv("AI_SDK", "a-sdk-that-does-not-exist")
    describer = describer_from_env(_env())

    assert describer.fault == "config:no_provider"
    with caplog.at_level(logging.ERROR, logger="basecradle_harness"):
        assert describer.describe_images([ImageContent(url="x", alt="a.png")]) is None
    line = next(
        r.getMessage()
        for r in caplog.records
        if "purpose=helper" in r.getMessage() and "outcome=fallback" in r.getMessage()
    )
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
    line = next(
        r.getMessage()
        for r in caplog.records
        if "purpose=helper" in r.getMessage() and "outcome=fallback" in r.getMessage()
    )
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
    assert note.index("Watched the whole of") < note.index(VIDEO_DESCRIPTION)


def test_a_described_clip_is_trimmed_to_the_window_and_the_line_says_which_seconds():
    """Issue #482 on the describer's own native path — the same `native_watch`, one seam.

    A blind brain has even less to go on than a sighted one: it cannot look and see whether it got
    the whole clip, so the facts line has to say. The probe reads the clip that was **sent**, so
    the duration is the window's, not the file's.
    """
    clip = _clip()
    clip.sampling = FrameSampling(start=1, end=2)
    describer = Describer(FakeDescriberProvider(video=True), "d/video-model")

    described = describer.describe_video(clip)

    assert described.startswith("(Watched 1s-2s of clip.mp4 (1.0s, 24 fps, 160x120).)")


def test_a_described_clip_names_a_window_the_native_tier_could_not_apply():
    """Issue #481's clause survives as #482's fallback — a trim that fails is never silent."""
    clip = VideoContent(
        url="data:video/mp4;base64,AAAA",
        alt="broken.mp4",
        sampling=FrameSampling(start=1, end=2),
    )
    describer = Describer(FakeDescriberProvider(video=True), "d/video-model")

    described = describer.describe_video(clip)

    # The header will not parse either, so the facts line drops out entirely and the description
    # stands alone — a probe failure is a fact about this decoder, not about the clip (#479).
    assert described == VIDEO_DESCRIPTION


def test_an_unprobeable_clip_costs_the_facts_line_and_never_the_description():
    """A header that will not parse is a fact about this decoder, not about the clip.

    The describer has already watched it and answered; dropping that answer over a probe — or
    replacing it with a note about the parse — would spend the valuable half to report the cheap
    one.
    """
    clip = VideoContent(url="data:video/mp4;base64,bm90YXZpZGVv", alt="broken.mp4")
    describer = Describer(FakeDescriberProvider(video=True), "d/video-model")

    assert describer.describe_video(clip) == VIDEO_DESCRIPTION


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
    assert VIDEO_DESCRIPTION in note.content


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

    line = next(
        r.getMessage()
        for r in caplog.records
        if "purpose=helper" in r.getMessage() and "outcome=fallback" in r.getMessage()
    )
    assert secret not in line
    assert "[redacted]" in line


# === One attempt, one line: the #485 grammar ==================================


class LoggingDescriberProvider(FakeDescriberProvider):
    """A describer adapter that logs its call the way every real one does.

    The plain fake never calls `log_llm_call`, so it cannot show the thing that matters here: a
    real adapter *does*, and the capture seam has to swallow that line so the caller can write the
    one line carrying the outcome. Without this the tests would pass on a describer whose adapter
    was silent — the one case where nothing needs suppressing.
    """

    def chat(self, messages, tools=None):
        from basecradle_harness._observability import log_llm_call

        log_llm_call(
            provider=self.provider,
            model=self.model,
            seconds=1.5,
            endpoint="Novita",
            cost=0.0021,
            usage={"prompt_tokens": 812, "completion_tokens": 96},
        )
        return super().chat(messages, tools)


def _helper_lines(caplog):
    return [r for r in caplog.records if r.getMessage().startswith("llm ")]


def test_a_describe_logs_exactly_one_llm_line_carrying_purpose_helper(caplog):
    """The grammar's own invariant: one `llm` line per attempt, and no other line carries its cost.

    Three things were wrong before issue #485 and this pins all three. The describer's call landed
    on a plain `llm` line **indistinguishable from the brain's**, so a second model's spend read as
    the first's; it also wrote a ` media provider=describer ` line, and that head is the
    dashboard's *tools* category, which a model call is not; and neither line carried an
    ``outcome=``, so a describer that was configured and **dead** looked exactly like a
    deliberately blind agent.
    """
    describer = Describer(LoggingDescriberProvider(), "d/model")

    with caplog.at_level(logging.DEBUG, logger="basecradle_harness"):
        assert describer.describe_images([ImageContent(url="x", alt="cat.png")]) == DESCRIPTION

    lines = _helper_lines(caplog)
    assert len(lines) == 1, [r.getMessage() for r in lines]
    message = lines[0].getMessage()
    assert lines[0].levelno == logging.INFO  # the feature working is not a warning
    for field in (
        "provider=openrouter",
        "purpose=helper",
        "kind=image.describe",
        "endpoint=Novita",  # the adapter's knowledge, carried onto the caller's line
        "model=d/model",
        "duration=1.50s",
        "tokens_in=812",
        "cost=0.0021",
        "outcome=ok",
        "subject=cat.png",
    ):
        assert field in message, message
    assert "reason=" not in message
    # The money is on exactly one line, and never on the tools head.
    assert not [r for r in caplog.records if "cost=" in r.getMessage() and r is not lines[0]]
    assert not [r for r in caplog.records if " media provider=" in f" {r.getMessage()}"]


def test_a_clip_is_described_under_the_video_kind():
    assert VIDEO_KIND == "video.describe" and IMAGE_KIND == "image.describe"


def test_an_answer_that_arrives_unusable_is_still_one_line_and_still_billed(caplog):
    """A call that answered with nothing usable was made and charged for — the reranker's rule.

    So its tokens and cost ride the same single line a success would write, with
    ``outcome=fallback``. Emitting the adapter's line *and* a failure line would count the attempt
    twice and put the dollar in the category twice over.
    """
    describer = Describer(LoggingDescriberProvider(answer="   "), "d/model")

    with caplog.at_level(logging.DEBUG, logger="basecradle_harness"):
        assert describer.describe_images([ImageContent(url="x", alt="cat.png")]) is None

    lines = _helper_lines(caplog)
    assert len(lines) == 1
    message = lines[0].getMessage()
    assert lines[0].levelno == logging.WARNING  # runtime-class: it can succeed next time
    assert "outcome=fallback" in message and "reason=empty_response" in message
    assert "cost=0.0021" in message  # billed, and counted exactly once


def test_a_call_that_never_happened_carries_no_duration_and_no_cost(caplog):
    """A config-class fault has no call behind it — so the line is honest about having none."""
    describer = describer_from_env(_env(**{DESCRIBER_API_KEY_VAR: None}))

    with caplog.at_level(logging.DEBUG, logger="basecradle_harness"):
        describer.describe_images([ImageContent(url="x", alt="cat.png")])

    message = _helper_lines(caplog)[0].getMessage()
    assert "outcome=fallback" in message and "reason=config:missing_api_key" in message
    assert "duration=" not in message and "cost=" not in message


class LoggingBlindProvider(BlindProvider):
    """The brain, logging its own calls the way every real adapter does."""

    def chat(self, messages, tools=None):
        from basecradle_harness._observability import log_llm_call

        log_llm_call(
            provider=self.provider,
            model=self.model,
            seconds=4.0,
            cost=0.05,
            usage={"prompt_tokens": 90210, "completion_tokens": 40},
        )
        return super().chat(messages, tools)


def test_a_wake_that_describes_tells_the_two_models_spend_apart(caplog):
    """The defect issue #485 closes, end to end: a second model's dollars read as the first's.

    @glm-5.2 is the fleet's only text-only brain, so on every picture it hands the pixels to a
    Gemini-class describer — a real OpenRouter call on the agent's own account. Both landed on a
    plain `llm` line with nothing to tell them apart, so the describer's spend inflated the brain's
    rollup and the helper had no cost series of its own to appear in.
    """
    brain = LoggingBlindProvider(*_turn("view"))
    engine = _engine(brain, ViewTool(), Describer(LoggingDescriberProvider(), "d/model"))

    with caplog.at_level(logging.DEBUG, logger="basecradle_harness"):
        engine.run([Message.user("look")])

    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("llm ")]
    main = [m for m in lines if "purpose=main" in m]
    helper = [m for m in lines if "purpose=helper" in m]
    assert len(main) == 2  # the brain's two turns
    assert len(helper) == 1  # one picture, one describe, one line
    assert len(main) + len(helper) == len(lines)  # every model call names a purpose
    # The dollars are separable, which is the whole point.
    assert all("cost=0.05" in m for m in main)
    assert "cost=0.0021" in helper[0] and "model=d/model" in helper[0]
    assert "z-ai/glm-5.2" not in helper[0]


# === issue #488: a fragment is never handed to the brain as sight ============

#: What a working call reports. Any non-zero count is a real measurement, which is exactly the
#: thing the live failure did not have.
USAGE = {"prompt_tokens": 812, "completion_tokens": 96}

#: The live shape of a stream that broke: an answer arrived and the vendor counted nothing.
NO_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

#: The first-frame paragraph, cut mid-word, that @glm-5.2 was handed as its sight of a 5 s clip.
CUT_OFF = "First frame: a poster on a wall; small black text reads \u201c#40\u201d above a"


class WireDescriberProvider(FakeDescriberProvider):
    """A describer adapter that reports what a real one reports — usage, cost, a finish reason.

    The plain fake reports none of it, which is the one case where nothing needs judging: every
    detection here is a judgement about what the *vendor said*, so it has to be said.
    """

    def __init__(self, *, finish=None, usage=USAGE, cost=0.0021, **kwargs):
        super().__init__(**kwargs)
        self.finish = finish
        self.usage = usage
        #: The dollar the vendor stated, if it stated one. A vendor that counted nothing usually
        #: charges nothing either — the live line carried neither — and the two are separate
        #: claims, so the fake lets a test say exactly which one it is making.
        self.cost = cost

    def chat(self, messages, tools=None):
        from basecradle_harness._observability import log_llm_call

        log_llm_call(
            provider=self.provider,
            model=self.model,
            seconds=1.5,
            usage=self.usage,
            cost=self.cost,
            finish_reason=self.finish,
        )
        return super().chat(messages, tools)


def _describer(*providers, model="d/model"):
    """A describer whose retry (or video) budget hands out the *next* provider, recording budgets.

    Mirrors production exactly: the still budget's adapter is built eagerly and seeded, and every
    other budget is built on the call that needs it.
    """
    asked = []
    queue = list(providers[1:])

    def build(budget):
        asked.append(budget)
        return queue.pop(0)

    return Describer(providers[0], model, build=build, budget=IMAGE_OUTPUT_BUDGET), asked


def _fallback_lines(caplog):
    return [r for r in _helper_lines(caplog) if "outcome=fallback" in r.getMessage()]


def _notes(caplog):
    """The `usage unreported` notes — deliberately **not** on the `llm` head (issue #491).

    A note is not a call record: it explains why one has a hole in it. Sharing the head would put
    it in every column that counts model calls, spend included.
    """
    return [r for r in caplog.records if r.getMessage().startswith("usage unreported ")]


def test_a_truncated_answer_is_never_handed_to_the_brain_as_sight(caplog):
    """The live defect (issue #488): 1,748 characters cut mid-sentence, logged ``outcome=ok``.

    The caption tells the brain it is reading a description of the whole thing, and a brain has no
    way to notice the sentence stopped in the middle — so a fragment presented as a whole one is a
    fabrication by omission, and the honest caption it replaces is strictly better.
    """
    vision = WireDescriberProvider(finish="length", answer=CUT_OFF)
    describer = Describer(vision, "d/model")

    with caplog.at_level(logging.DEBUG, logger="basecradle_harness"):
        assert describer.describe_images([ImageContent(url="x", alt="poster.png")]) is None

    lines = _fallback_lines(caplog)
    assert len(lines) == 1
    assert "reason=truncated" in lines[0].getMessage()
    assert lines[0].levelno == logging.WARNING  # runtime-class: it can succeed next time


def test_a_truncated_answer_is_retried_once_with_more_room(caplog):
    """A vendor that stopped at ``length`` ran out of room, which a bigger budget is the fix for.

    So the describe is two attempts, not one — and each writes its own `llm` line, because each is
    a call that was made and billed (the #485 grammar, not a duplicate).
    """
    describer, asked = _describer(
        WireDescriberProvider(finish="length", answer=CUT_OFF),
        WireDescriberProvider(answer=DESCRIPTION),
    )

    with caplog.at_level(logging.DEBUG, logger="basecradle_harness"):
        assert describer.describe_images([ImageContent(url="x", alt="poster.png")]) == DESCRIPTION

    assert asked == [IMAGE_OUTPUT_BUDGET * RETRY_BUDGET_FACTOR]
    lines = _helper_lines(caplog)
    assert len(lines) == 2
    assert "reason=truncated" in lines[0].getMessage()
    assert "outcome=ok" in lines[1].getMessage() and lines[1].levelno == logging.INFO


def test_a_second_truncation_falls_back_rather_than_spending_again(caplog):
    """Once. A second ``length`` is the model saying the budget was never the problem."""
    describer, asked = _describer(
        WireDescriberProvider(finish="length", answer=CUT_OFF),
        WireDescriberProvider(finish="length", answer=CUT_OFF),
        WireDescriberProvider(answer=DESCRIPTION),  # never reached
    )

    with caplog.at_level(logging.DEBUG, logger="basecradle_harness"):
        assert describer.describe_images([ImageContent(url="x", alt="poster.png")]) is None

    assert len(asked) == 1  # one retry, one extra adapter
    assert [r.getMessage().count("reason=truncated") for r in _helper_lines(caplog)] == [1, 1]


def test_an_answer_the_vendor_counted_nothing_for_is_still_an_answer(caplog):
    """Issue #491, and the rule #488 got backwards: absent usage is a fact about the **bill**.

    Live, on 0.118.2: OpenRouter/Google report no usage at all for ``google/gemini-3.8-flash``'s
    *video* calls — while the same model reports it for images in the same wake, and the older
    ``gemini-3.1-flash-lite`` reported it for video. Complete, three-part descriptions were being
    discarded over a vendor's bookkeeping. A vendor that did not count the call has said nothing
    about the text it returned.
    """
    vision = WireDescriberProvider(usage=NO_USAGE, cost=None, answer=DESCRIPTION)
    describer = Describer(vision, "d/model")

    with caplog.at_level(logging.DEBUG, logger="basecradle_harness"):
        assert describer.describe_images([ImageContent(url="x", alt="poster.png")]) == DESCRIPTION

    message = _helper_lines(caplog)[0].getMessage()
    assert "outcome=ok" in message and "reason=" not in message
    # Honest absence, both halves: no zeros dressed up as measurements, and no invented dollar.
    assert "tokens_" not in message and "cost=" not in message


def test_a_video_whose_usage_is_unreported_is_judged_on_its_parts_alone(caplog):
    """The live cell exactly: a complete three-part clip description, and no usage beside it.

    Both arms of the live re-run — with a `start`/`end` window and without — failed identically, so
    the trim was never the cause. What is judged is the answer: its finish reason, its text, its
    three parts.
    """
    vision = WireDescriberProvider(video=True, usage=NO_USAGE, answer=VIDEO_DESCRIPTION)
    describer = Describer(vision, "d/video-model")

    with caplog.at_level(logging.DEBUG, logger="basecradle_harness"):
        assert VIDEO_DESCRIPTION in (describer.describe_video(_clip()) or "")

    assert _fallback_lines(caplog) == []


def test_the_unreported_bill_is_noted_once_per_wake_for_that_model_and_kind(caplog):
    """A gap nobody can explain gets explained wrongly — which is how #488 happened.

    So the hole in the helper cost series says why it is there: one INFO note naming the cell, the
    first time that cell reports nothing. Once per wake, because the object's life *is* the wake;
    per model+kind, because the vendor's accounting is a property of the cell — this very model
    counts an ``image.describe`` and not a ``video.describe``.
    """
    vision = WireDescriberProvider(usage=NO_USAGE, cost=None, answer=DESCRIPTION)
    describer = Describer(vision, "d/model")

    with caplog.at_level(logging.DEBUG, logger="basecradle_harness"):
        for _ in range(3):
            describer.describe_images([ImageContent(url="x", alt="poster.png")])

    notes = _notes(caplog)
    assert len(notes) == 1
    assert notes[0].levelno == logging.INFO  # a fact, not a fault
    message = notes[0].getMessage()
    for field in ("provider=openrouter", "purpose=helper", "kind=image.describe", "model=d/model"):
        assert field in message, message
    # It explains a *missing* series; it must never look like a call record with zeros in it, and
    # it is not one — three describes are three `llm` lines however many notes they earned, so the
    # spend partition over that head counts exactly the calls that were made.
    assert "tokens_" not in message and "cost=" not in message and "outcome=" not in message
    assert not message.startswith("llm ") and len(_helper_lines(caplog)) == 3


def test_a_vendor_that_states_no_usage_at_all_is_not_noted(caplog):
    """The `usage_reported` line, kept: silence about a claim is not the claim.

    An adapter that never instrumented — a surface that omits usage, a third-party `Provider` —
    has told us nothing, and a note about a claim nobody made is noise on every wake forever.
    """
    describer = Describer(FakeDescriberProvider(answer=DESCRIPTION), "d/model")

    with caplog.at_level(logging.DEBUG, logger="basecradle_harness"):
        assert describer.describe_images([ImageContent(url="x", alt="poster.png")]) == DESCRIPTION

    assert _notes(caplog) == []


def test_a_video_description_missing_its_parts_is_not_a_video_description(caplog):
    """Since #479 the three parts *are* the contract, and #488 is the check that was missing.

    The live answer had `First frame:` and nothing after it. The harness cannot tell a clip
    described in one paragraph from a clip whose description stopped after the first frame — and it
    must not guess in the direction of "close enough".
    """
    vision = WireDescriberProvider(video=True, answer="The entire image vibrates throughout.")
    describer = Describer(vision, "d/video-model")

    with caplog.at_level(logging.DEBUG, logger="basecradle_harness"):
        assert describer.describe_video(_clip()) is None

    assert "reason=missing_parts" in _fallback_lines(caplog)[0].getMessage()


@pytest.mark.parametrize(
    "answer",
    [
        "**First frame:** a cat\n**Over time:** it moves\n**Last frame:** it is gone",
        "## First frame: a cat\n## Over time: it moves\n## Last frame: it is gone",
        "- first frame: a cat\n- over time: it moves\n- last frame: it is gone",
        "First frame: a cat. Over time: it moves. Last frame: it is gone.",
    ],
)
def test_a_describer_that_answers_in_bold_is_still_answering(answer):
    """The check catches a **missing part**; it does not police layout.

    A model told "plain prose, no markdown headings" will sometimes use them anyway, and will
    sometimes write the three labels inline in one paragraph — which is exactly what "plain prose"
    invites. Discarding a complete description over a pair of asterisks or a line break would blind
    the agent to enforce a style nobody reads.
    """
    describer = Describer(WireDescriberProvider(video=True, answer=answer), "d/video-model")

    assert describer.describe_video(_clip()) is not None


def test_a_still_is_never_judged_against_the_video_parts():
    """A photograph has no first frame, so the check that guards a clip must not reach a picture."""
    describer = Describer(WireDescriberProvider(answer=DESCRIPTION), "d/model")

    assert describer.describe_images([ImageContent(url="x", alt="cat.png")]) == DESCRIPTION


def test_a_clip_gets_three_parts_worth_of_room_and_a_still_gets_one():
    """The budget is a bound on both ends: enough room for what was asked, and a cap on what the
    transcript keeps — a description rides an injected turn, which is persisted for the timeline's
    life (Context Discipline)."""
    assert VIDEO_OUTPUT_BUDGET == 3 * IMAGE_OUTPUT_BUDGET

    describer, asked = _describer(
        WireDescriberProvider(video=True),
        WireDescriberProvider(video=True),
    )
    describer.describe_video(_clip())

    assert asked == [VIDEO_OUTPUT_BUDGET]


def test_a_describer_with_no_builder_describes_and_never_retries_itself(caplog):
    """The regression bar for the budget machinery: a library caller's own `Provider` still works.

    Every shipped adapter takes its cap at **construction**, so a directly-built describer has no
    larger budget to reach for. Asking it the identical question a second time would buy the
    identical answer, so it does not — the retry is conditioned on there being more room, never on
    the fault alone.
    """
    working = Describer(WireDescriberProvider(), "d/model")
    stuck = Describer(WireDescriberProvider(finish="length", answer=CUT_OFF), "d/model")

    with caplog.at_level(logging.DEBUG, logger="basecradle_harness"):
        assert working.describe_images([ImageContent(url="x", alt="cat.png")]) == DESCRIPTION
        caplog.clear()
        assert stuck.describe_images([ImageContent(url="x", alt="poster.png")]) is None

    assert len(_helper_lines(caplog)) == 1


def test_a_helper_line_never_says_a_free_successful_call_that_measured_nothing(caplog):
    """The contradiction the live line carried, stated as the invariant it violates (issue #488).

    ``outcome=ok tokens_in=0 … cost=0`` reads on a dashboard as a free call that worked, and a
    zero is not a measurement: whatever the outcome, an unreported count is **omitted** rather than
    printed. Note that ``outcome=ok`` beside *no* token fields is entirely legitimate (issue #491)
    — some vendors simply do not count some calls — which is why what is pinned here is the pairing
    of a verdict with a **zero**, never the verdict itself.
    """
    for vision in (
        WireDescriberProvider(),
        WireDescriberProvider(usage=NO_USAGE, answer=CUT_OFF),
        WireDescriberProvider(finish="length", answer=CUT_OFF),
        WireDescriberProvider(answer="   "),
    ):
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="basecradle_harness"):
            Describer(vision, "d/model").describe_images([ImageContent(url="x", alt="a.png")])
        for record in _helper_lines(caplog):
            message = record.getMessage()
            assert not ("outcome=ok" in message and "tokens_in=0" in message), message
            assert not ("outcome=ok" in message and "cost=0 " in f"{message} "), message
