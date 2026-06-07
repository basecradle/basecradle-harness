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

    def to_dict(self) -> dict[str, Any]:
        """A JSON-safe mapping of this turn, for persisting a session transcript.

        Only the fields that carry meaning for this role are emitted, so a stored
        transcript reads cleanly: a plain user turn is just `{role, content}`.
        """
        data: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            data["content"] = self.content
        if self.tool_calls:
            data["tool_calls"] = [
                {"id": c.id, "name": c.name, "arguments": c.arguments} for c in self.tool_calls
            ]
        if self.tool_call_id is not None:
            data["tool_call_id"] = self.tool_call_id
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        """Rebuild a `Message` from `to_dict` output — the read side of persistence."""
        return cls(
            role=data["role"],
            content=data.get("content"),
            tool_calls=[
                ToolCall(id=c["id"], name=c["name"], arguments=c.get("arguments", {}))
                for c in data.get("tool_calls", [])
            ],
            tool_call_id=data.get("tool_call_id"),
        )


@dataclass
class ToolSpec:
    """A tool offered to the model, in provider-neutral form.

    `parameters` is a JSON-Schema object describing the call arguments. Each
    provider adapter serializes this into its own tool format.
    """

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
