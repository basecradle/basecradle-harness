"""The transport-free OpenAI-wire serializers (`_openai_wire`), tested as pure functions.

These functions translate the harness's provider-agnostic `Message` vocabulary to and from the
OpenAI wire (Chat Completions and Responses), independent of *how* a request is carried. The
end-to-end path — the same dicts marshalled by a real vendor SDK — is exercised in `test_provider`
(the ``openai`` adapter's ``chat`` surface) and `test_openrouter` (the native OpenRouter SDK); the
cache-breakpoint shape lives in `test_caching`. This file pins the shapes those callers depend on.

The Chat Completions image serialization is the focus (issue #313): the ``chat`` surface used to
drop ``message.images`` on the wire while the Responses surface (`_input_content`) and the native
xai-sdk adapter serialized them, so a vision-capable model reached over Chat Completions could not
see a posted image even though the model could. These tests pin the vision content-part shape and
its edges.
"""

from basecradle_harness import ImageContent, Message
from basecradle_harness._openai_wire import chat_message_to_wire

DATA_URL = "data:image/png;base64,AAAA"


def _vision_turn(content, *images):
    turn = Message(role="user", content=content)
    turn.images = list(images)
    return turn


def test_a_vision_turn_serializes_text_then_the_image_part():
    """The text leads as a ``text`` part, then the image as a nested ``image_url`` object — Chat
    Completions nests the reference (``image_url.url``) where Responses uses the bare string."""
    wire = chat_message_to_wire(_vision_turn("what's this?", ImageContent(url=DATA_URL, alt="c")))

    assert wire["role"] == "user"
    assert wire["content"] == [
        {"type": "text", "text": "what's this?"},
        {"type": "image_url", "image_url": {"url": DATA_URL}},
    ]


def test_a_vision_turn_with_no_text_omits_the_empty_text_part():
    """An image-only turn (no caption) is just the image part — never a hollow ``text: ""`` part,
    which some endpoints reject."""
    wire = chat_message_to_wire(_vision_turn(None, ImageContent(url=DATA_URL, alt="c")))

    assert wire["content"] == [{"type": "image_url", "image_url": {"url": DATA_URL}}]


def test_multiple_images_each_become_an_image_url_part_in_order():
    """Every image is serialized — none dropped — in the order the turn carries them, after the
    text part."""
    first = "data:image/png;base64,AAAA"
    second = "data:image/jpeg;base64,BBBB"
    wire = chat_message_to_wire(
        _vision_turn("two:", ImageContent(url=first), ImageContent(url=second))
    )

    assert wire["content"] == [
        {"type": "text", "text": "two:"},
        {"type": "image_url", "image_url": {"url": first}},
        {"type": "image_url", "image_url": {"url": second}},
    ]


def test_an_anchored_vision_turn_keeps_the_breakpoint_on_the_text_part():
    """A cache breakpoint (`cache_anchor`, an explicit-cache vendor only) rides the leading text
    part even when the turn also carries an image, so a multimodal turn can still close the cacheable
    prefix. The two never coincide in practice, but the breakpoint must not be silently lost if they
    do."""
    turn = _vision_turn("frozen", ImageContent(url=DATA_URL))
    turn.cache_anchor = True

    wire = chat_message_to_wire(turn)

    assert wire["content"] == [
        {"type": "text", "text": "frozen", "cache_control": {"type": "ephemeral"}},
        {"type": "image_url", "image_url": {"url": DATA_URL}},
    ]


def test_a_turn_without_images_is_unchanged():
    """The image branch is guarded on ``message.images`` — a text turn stays a bare string, so no
    existing turn changes shape and no endpoint sees a parts list it didn't before."""
    assert chat_message_to_wire(Message.user("hello"))["content"] == "hello"
