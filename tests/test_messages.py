"""The normalized message vocabulary: constructors and defaults."""

from basecradle_harness import ImageContent, Message, ToolCall, ToolResult, ToolSpec


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
