"""The normalized message vocabulary: constructors and defaults."""

from basecradle_harness import Message, ToolCall, ToolSpec


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
