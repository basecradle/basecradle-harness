"""The agent loop: think → act → think → … → respond.

The engine is the nervous system, and it is deliberately ignorant of "safe". It
holds no policy of its own: it runs whatever tools its `ToolRegistry` contains,
and that registry is what a policy gated at registration time. Hand it a locked
registry and it is Harness; hand it an unlocked one and the very same loop is
Cradle. That is the whole "one core, two profiles" design, and it is why there
is not a single Harness-specific assumption in this file.

One turn (`run`) is: ask the provider for the next message; if it is plain text,
that is the reply; if it carries tool calls, run each through the registry,
append the results, and ask again — until the model answers with no more calls
or the step limit is hit.

A tool may return more than text. When it returns a `ToolResult` carrying images
(the assets tool's `view` action does, so a peer can *see* a picture), the engine
appends the text as the `tool` result and then injects the images as a synthetic
`user` turn — because on every provider a function-tool *result* is text-only;
an image has to enter as model *input*. Once the model has answered, the engine
**evicts** those pixels (keeping a short text breadcrumb), so a viewed image is
not re-sent — and re-billed — on every later turn. Viewing is on-demand: cheap to
do again, never a standing cost.
"""

from __future__ import annotations

from basecradle_harness._exceptions import EngineError
from basecradle_harness._messages import ImageContent, Message, ToolResult
from basecradle_harness._provider import Provider
from basecradle_harness._tools import ToolRegistry

DEFAULT_MAX_STEPS = 8


class Engine:
    """Runs the think→act loop for one provider against one tool registry.

    Args:
        provider: The model. Only its `chat` method is used.
        tools: The registry whose tools the model may call. Its policy has
            already gated what could be registered; the engine just runs them.
        max_steps: The most provider calls one `run` may make before giving up.
            Bounds runaway tool loops.
    """

    def __init__(
        self, provider: Provider, tools: ToolRegistry, *, max_steps: int = DEFAULT_MAX_STEPS
    ) -> None:
        self.provider = provider
        self.tools = tools
        self.max_steps = max_steps

    def run(self, messages: list[Message]) -> Message:
        """Drive the conversation to a final text reply.

        Appends each assistant turn and every tool result onto `messages` (so the
        list is the full transcript afterward) and returns the final assistant
        message. Raises `EngineError` if no final reply arrives within `max_steps`.
        """
        specs = self.tools.specs() or None
        shown: list[Message] = []  # image turns injected this run, evicted before returning
        # The eviction must happen however the loop ends — including the max_steps
        # error path — or a viewed image's base64 lingers in the (mutated-in-place)
        # transcript and is re-sent on every later turn, the cost this exists to avoid.
        try:
            for _ in range(self.max_steps):
                reply = self.provider.chat(messages, tools=specs)
                messages.append(reply)
                if not reply.tool_calls:
                    return reply
                for call in reply.tool_calls:
                    result = self._run_tool(call.name, call.arguments)
                    text, images = _split_result(result)
                    messages.append(Message.tool(tool_call_id=call.id, content=text))
                    if images:
                        # A function-tool result is text-only on every provider, so an
                        # image enters as model *input*: a synthetic user turn carrying it.
                        shown_turn = Message(role="user", content=_caption(images), images=images)
                        messages.append(shown_turn)
                        shown.append(shown_turn)
            raise EngineError(
                f"No final reply after {self.max_steps} steps; the model kept calling tools."
            )
        finally:
            _evict_images(shown)

    def _run_tool(self, name: str, arguments: dict) -> str | ToolResult:
        """Run one tool call, turning any failure into a result the model can read.

        Errors are fed back as the tool's output rather than raised: a model that
        called a missing tool or passed bad arguments can see what went wrong and
        try again, which is how a real agent recovers.
        """
        try:
            tool = self.tools.get(name)
        except KeyError:
            return f"Error: no tool named {name!r}."
        try:
            return tool.run(**arguments)
        except Exception as exc:  # noqa: BLE001 - any tool failure becomes model-readable
            return f"Error running {name!r}: {exc}"


def _split_result(result: str | ToolResult) -> tuple[str, list[ImageContent]]:
    """Normalize a tool's return into (text, images) — a plain `str` has no images."""
    if isinstance(result, ToolResult):
        return result.text, result.images
    return result, []


def _caption(images: list[ImageContent]) -> str:
    """A one-line caption for an injected image turn, naming the images shown.

    It rides along as the user turn's text, so a provider that cannot render
    images still sees *what* was shared, and it stays as the breadcrumb after the
    pixels are evicted.
    """
    names = ", ".join(image.alt or "image" for image in images)
    return f"(Showing image: {names})"


def _evict_images(shown: list[Message]) -> None:
    """Drop the pixels from injected image turns once the model has answered.

    The model has already seen the image and folded it into its reply, so the raw
    bytes need not persist into the transcript — keeping them would re-send and
    re-bill the image on every later turn. The text caption stays as a breadcrumb,
    so the conversation still reads coherently; viewing again is a fresh, bounded,
    on-demand fetch.
    """
    for turn in shown:
        turn.images = []
