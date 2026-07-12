"""The brain contract.

A `Provider` is the model behind an agent. The contract is deliberately tiny:
one `chat` call that takes the conversation so far and the tools on offer, and
returns the model's next turn. Everything provider-specific — endpoints, auth,
payload shapes, tool-call encoding — lives inside an adapter.

Adding a provider is implementing this one method. That is the whole promise:
the engine depends on `Provider`, never on any concrete adapter.

**Capabilities: what an adapter may also answer, and what happens when it can't.** Beyond `chat`,
the harness asks an adapter a few optional questions and reads them by capability — never by a
vendor branch, and never fatally. Two belong to the context budget (issue #276):

- **`last_tokens_in: int | None`** — the input-token count the endpoint reported for the adapter's
  most recent call. Every shipped adapter records it beside the usage it already logs, and it is the
  compaction trigger: exact, free, and needing no tokenizer (which matters — GLM publishes none, so
  a client-side count could not be honest).
- **`context_limit() -> int | None`** — this model's context ceiling, answered however the adapter
  honestly can, and ``None`` when it cannot. xAI reads its SDK's ``max_prompt_length``; OpenRouter
  computes the live ceiling of the endpoints it would actually route to; OpenAI states no context
  window anywhere and so answers ``None``. There is deliberately **no static model→limit table** —
  it cannot express a router's reality and it rots silently.

An adapter that implements neither still works: the budget falls back to a conservative floor and,
with no usage to read, simply never triggers compaction. A capability is a question, not a contract.
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
