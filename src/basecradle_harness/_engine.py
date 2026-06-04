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
"""

from __future__ import annotations

from basecradle_harness._exceptions import EngineError
from basecradle_harness._messages import Message
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
        for _ in range(self.max_steps):
            reply = self.provider.chat(messages, tools=specs)
            messages.append(reply)
            if not reply.tool_calls:
                return reply
            for call in reply.tool_calls:
                result = self._run_tool(call.name, call.arguments)
                messages.append(Message.tool(tool_call_id=call.id, content=result))
        raise EngineError(
            f"No final reply after {self.max_steps} steps; the model kept calling tools."
        )

    def _run_tool(self, name: str, arguments: dict) -> str:
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
