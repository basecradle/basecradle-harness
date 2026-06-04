"""The brain contract.

A `Provider` is the model behind an agent. The contract is deliberately tiny:
one `chat` call that takes the conversation so far and the tools on offer, and
returns the model's next turn. Everything provider-specific — endpoints, auth,
payload shapes, tool-call encoding — lives inside an adapter.

Adding a provider is implementing this one method. That is the whole promise:
the engine depends on `Provider`, never on any concrete adapter.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from basecradle_harness._messages import Message, ToolSpec


@runtime_checkable
class Provider(Protocol):
    """A model that can hold a conversation and call tools.

    `chat` is given the full message history and the tools available this turn,
    and returns the model's next message — which may carry text, `tool_calls`,
    or both. The engine runs any tool calls, appends their results as `tool`
    messages, and calls `chat` again until the model answers with no more calls.
    """

    def chat(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec] | None = None
    ) -> Message: ...
