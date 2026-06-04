"""The normalized, provider-agnostic vocabulary the engine speaks.

A `Provider` adapter translates between these types and its own wire format, so
nothing above the provider layer ever sees an OpenAI (or xAI, or OpenRouter)
payload. Three small dataclasses are the whole vocabulary:

- `Message` — one turn in the conversation (system / user / assistant / tool).
- `ToolCall` — the model asking to run a tool, with `arguments` already parsed
  into a `dict` (never a JSON string — that is a wire detail the adapter owns).
- `ToolSpec` — a tool offered to the model: a name, a description, and a
  JSON-Schema description of its parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ToolCall:
    """A model's request to invoke one tool.

    `arguments` is the decoded mapping, not the JSON string the wire carries —
    callers work with data, not serialization.
    """

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class Message:
    """One turn in a conversation.

    `content` is the text (absent on an assistant turn that is purely tool
    calls). `tool_calls` is populated only on assistant turns. `tool_call_id`
    is set only on a `tool` turn, linking a result back to the call it answers.
    """

    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None

    @classmethod
    def system(cls, content: str) -> Message:
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(role="user", content=content)

    @classmethod
    def assistant(
        cls, content: str | None = None, tool_calls: list[ToolCall] | None = None
    ) -> Message:
        return cls(role="assistant", content=content, tool_calls=tool_calls or [])

    @classmethod
    def tool(cls, tool_call_id: str, content: str) -> Message:
        """A tool's result, answering the call with id `tool_call_id`."""
        return cls(role="tool", content=content, tool_call_id=tool_call_id)


@dataclass
class ToolSpec:
    """A tool offered to the model, in provider-neutral form.

    `parameters` is a JSON-Schema object describing the call arguments. Each
    provider adapter serializes this into its own tool format.
    """

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
