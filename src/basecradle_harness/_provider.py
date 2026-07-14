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

A third belongs to prompt caching (issue #277):

- **`cache_mode: "automatic" | "explicit" | "none"`** — how this adapter's endpoint reaches its
  prompt cache (`_caching`). ``automatic`` (OpenAI, xAI, OpenRouter today) and ``none`` mean the
  engine does nothing; ``explicit`` (Anthropic) means the client must *mark* the cacheable prefix or
  there is no caching **at all**, and the engine places one breakpoint at the stable/volatile
  boundary of the message list it already maintains. An adapter that declares nothing resolves to
  ``automatic`` — the do-nothing default — so an adapter written before this capability existed is
  unaffected.

  **This one is a standing rule, not just a capability: no new provider adapter ships without
  declaring its `cache_mode`** (`CLAUDE.md` → Provider Capabilities). The other capabilities fail
  *safe* when unanswered — a missing `context_limit` costs a conservative floor, and a missing
  `last_tokens_in` costs a compaction that never fires. A missing `cache_mode` on an explicit-cache
  vendor fails **expensive and silent**: nothing errors, no log line changes, and the agent simply
  pays full freight on every token of every wake, forever. That asymmetry is why this question must
  be answered *before* the first agent on a new provider is provisioned, not after its first bill.

A fourth belongs to perception (issue #228):

- **`supports_vision() -> bool | None`** — whether the configured model accepts image input. The
  asset-wake reads it before showing a peer's posted image to the model: ``True`` shows the image
  as before, ``False`` swaps it for its text description (a text-only model would otherwise be shown
  pixels it silently drops or an endpoint rejects), and ``None`` means *unknown*. Only OpenRouter
  implements it today, from the model's own ``architecture.input_modalities``; the OpenAI and xAI
  adapters answer nothing (every model they run is vision-capable, and per-model modality is not
  cheaply readable there). The gate **fails open** — absent, ``None``, or a raise all read as "show
  the image" — so a missing capability or a metadata hiccup never withholds an image from a
  vision-capable agent; only a *definite* ``False`` degrades. A text-only model on an OpenAI/xAI
  surface is thus the future gap that owns implementing this for its adapter, the same way an
  explicit-cache vendor owns `cache_mode`.

An adapter that implements none of them still works: the budget falls back to a conservative floor,
with no usage to read it never triggers compaction, nothing is placed on the wire, and every image
is shown. A capability is a question, not a contract.
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
