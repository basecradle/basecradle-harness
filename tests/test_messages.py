"""The normalized message vocabulary: constructors and defaults."""

from basecradle_harness import (
    FrameSampling,
    ImageContent,
    Message,
    ToolCall,
    ToolResult,
    ToolSpec,
    VideoContent,
)


def test_role_constructors_set_the_role():
    assert Message.system("be helpful").role == "system"
    assert Message.user("hi").role == "user"
    assert Message.assistant("hello").role == "assistant"
    assert Message.tool(tool_call_id="call_1", content="42").role == "tool"


def test_user_message_has_no_tool_calls_by_default():
    msg = Message.user("hi")
    assert msg.content == "hi"
    assert msg.tool_calls == []
    assert msg.tool_call_id is None


def test_assistant_can_carry_tool_calls():
    call = ToolCall(id="call_1", name="search", arguments={"q": "peers"})
    msg = Message.assistant(tool_calls=[call])
    assert msg.content is None
    assert msg.tool_calls == [call]


def test_tool_message_links_to_its_call():
    msg = Message.tool(tool_call_id="call_1", content="result")
    assert msg.tool_call_id == "call_1"
    assert msg.content == "result"


def test_tool_call_arguments_default_to_empty_dict():
    assert ToolCall(id="c", name="noop").arguments == {}


def test_tool_spec_holds_json_schema_parameters():
    spec = ToolSpec(
        name="search",
        description="Search the timeline.",
        parameters={"type": "object", "properties": {"q": {"type": "string"}}},
    )
    assert spec.name == "search"
    assert spec.parameters["properties"]["q"]["type"] == "string"


# --- images & tool results ---------------------------------------------------


def test_message_carries_no_images_by_default():
    assert Message.user("hi").images == []


def test_tool_result_defaults_to_no_images():
    result = ToolResult(text="done")
    assert result.text == "done"
    assert result.images == []


def test_message_with_images_round_trips_through_dict():
    original = Message(
        role="user",
        content="(Showing image: cat.png)",
        images=[ImageContent(url="data:image/png;base64,AAAA", alt="cat.png")],
    )
    restored = Message.from_dict(original.to_dict())
    assert restored.content == "(Showing image: cat.png)"
    assert restored.images == [ImageContent(url="data:image/png;base64,AAAA", alt="cat.png")]


def test_to_dict_omits_images_when_there_are_none():
    # A plain turn stays clean — no empty `images` key cluttering the transcript.
    assert "images" not in Message.user("hi").to_dict()


# --- videos (issue #471) ------------------------------------------------------


def test_message_and_tool_result_carry_no_videos_by_default():
    assert Message.user("hi").videos == []
    assert ToolResult(text="done").videos == []


def test_frame_sampling_defaults_to_one_frame_a_second_over_the_whole_clip():
    sampling = FrameSampling()
    assert (sampling.every, sampling.start, sampling.end) == (1.0, None, None)


def test_message_with_videos_round_trips_through_dict():
    """A save can land *mid-turn*, before eviction, so a video turn has to survive the disk."""
    original = Message(
        role="user",
        content="(Showing video: clip.mp4)",
        videos=[
            VideoContent(
                url="data:video/mp4;base64,AAAA",
                alt="clip.mp4",
                content_type="video/mp4",
                sampling=FrameSampling(every=2, start=1, end=4),
            )
        ],
        injected=True,
    )

    restored = Message.from_dict(original.to_dict())

    assert restored.videos == original.videos
    assert restored.injected is True


def test_a_video_turn_restores_default_sampling_when_the_record_predates_it():
    """An older transcript has no `sampling` key — it reads back as the defaults, never a crash."""
    restored = Message.from_dict(
        {"role": "user", "videos": [{"url": "data:video/mp4;base64,AAAA", "alt": "clip.mp4"}]}
    )

    assert restored.videos[0].sampling == FrameSampling()
    assert restored.videos[0].content_type == "video/mp4"


def test_to_dict_omits_videos_when_there_are_none():
    assert "videos" not in Message.user("hi").to_dict()
    assert "videos" not in Message(role="user", images=[ImageContent(url="x")]).to_dict()
