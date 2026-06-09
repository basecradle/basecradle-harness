"""A second provider adapter: OpenAI's Responses API, for built-in server tools.

`OpenAICompatibleProvider` speaks Chat Completions and is portable across OpenAI,
xAI, and OpenRouter. This adapter speaks OpenAI's **Responses API**
(`POST /v1/responses`) instead — same `Provider` contract, same `chat` method,
different wire. Its reason to exist is the one thing Chat Completions cannot do:
**built-in, server-side tools**. The flagship is `web_search` — OpenAI runs the
search *inside* the API call and returns the model's answer already grounded in
live sources, with `url_citation` annotations. The harness never executes it.

The adapter is deliberately OpenAI-only (Responses is an OpenAI API), and it is
purely additive — `OpenAICompatibleProvider` stays the default; an agent opts
into this one when it wants web search (and, later, other built-ins).

Two kinds of tool coexist in one turn, and the split is the whole design:

- **Built-in tools** (`web_search`, and later `image_generation`, …) are declared
  by type and resolved *server-side*. Their output items are informational — the
  adapter does not turn them into a `ToolCall` for the engine to run.
- **Custom function tools** (the platform tools + memory) are declared with their
  JSON schema and still loop through the harness: a Responses turn may return a
  `function_call` the engine must execute and feed back, exactly as before.

So the response mapping has one rule that generalizes to any future built-in:
**only `message` items become reply text (plus citations); only `function_call`
items become `ToolCall`s; every other output item is server-side and ignored.**
Adding a built-in is registering its type in `builtin_tools` — never a rewrite.

Like the Chat Completions adapter it is **stateless per turn**: the full
conversation is sent every call and the harness owns history, so Responses'
server-side state (`previous_response_id`) is deliberately unused.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from typing import Any

import httpx

from basecradle_harness._exceptions import ProviderConnectionError, ProviderError
from basecradle_harness._http import raise_for_status
from basecradle_harness._messages import Message, ToolCall, ToolSpec

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_TIMEOUT = 60.0

#: The built-in server-side tools enabled by default. `web_search` is the reason
#: this adapter exists; an agent that wants others passes its own tuple.
DEFAULT_BUILTIN_TOOLS: tuple[str, ...] = ("web_search",)


class OpenAIResponsesProvider:
    """A `Provider` backed by OpenAI's Responses API, with built-in tools enabled.

    Satisfies the same `Provider` protocol as `OpenAICompatibleProvider` — the
    engine cannot tell them apart — but talks to `/responses` and lets OpenAI run
    server-side tools like `web_search` alongside the agent's custom function
    tools.

    Args:
        model: The model id (e.g. ``"gpt-5.4-mini"``). Must be a model that
            supports the Responses API and `web_search` (the GPT-5 series does).
        api_key: The OpenAI bearer token. Falls back to ``AI_PROVIDER_API_KEY``.
        base_url: The API root. Defaults to OpenAI; Responses is OpenAI-only, so
            this changes only for a proxy, not to reach another vendor.
        timeout: Per-request timeout in seconds.
        builtin_tools: The server-side built-ins to enable, as type names
            (``"web_search"``) or full tool dicts (for a built-in that needs
            configuration). Defaults to ``("web_search",)``. Each turn these are
            merged with the custom function tools the engine offers.
        default_params: Extra body parameters sent on every call (e.g.
            ``temperature=0.2``). ``model``, ``input``, and ``tools`` always take
            precedence and cannot be overridden here.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        builtin_tools: Sequence[str | Mapping[str, Any]] = DEFAULT_BUILTIN_TOOLS,
        **default_params: Any,
    ) -> None:
        key = api_key or os.environ.get("AI_PROVIDER_API_KEY")
        if not key:
            raise ValueError(
                "No API key: pass api_key=... or set the AI_PROVIDER_API_KEY environment variable."
            )
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._builtin_tools = [_builtin_to_wire(spec) for spec in builtin_tools]
        self._default_params = default_params
        self._client = httpx.Client(
            headers={"Authorization": f"Bearer {key}"},
            timeout=timeout,
        )

    def chat(self, messages: Sequence[Message], tools: Sequence[ToolSpec] | None = None) -> Message:
        """Run one Responses turn and return the model's reply (text and/or tool calls).

        The reply's `content` is the model's answer, with a `Sources:` footer
        appended when `web_search` cited any source. Its `tool_calls` are the
        *custom* function calls the engine must still run — web search is already
        resolved server-side and never appears there.
        """
        payload: dict[str, Any] = dict(self._default_params)
        payload["model"] = self.model
        payload["input"] = [item for m in messages for item in _message_to_input(m)]
        wire_tools = list(self._builtin_tools)
        if tools:
            wire_tools.extend(_function_tool_to_wire(t) for t in tools)
        if wire_tools:
            payload["tools"] = wire_tools

        try:
            response = self._client.post(f"{self.base_url}/responses", json=payload)
        except httpx.RequestError as exc:
            raise ProviderConnectionError(f"Could not reach the provider: {exc}") from exc

        if response.status_code >= 400:
            raise_for_status(response)

        return _message_from_wire(response.json())

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> OpenAIResponsesProvider:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# --- request: harness Message → Responses input items ------------------------


def _message_to_input(message: Message) -> list[dict[str, Any]]:
    """Serialize one `Message` into the Responses `input` items it maps to.

    Most turns are a single item; an assistant turn that both spoke and called
    tools becomes a text message *and* one `function_call` item per call, and a
    `tool` result becomes a `function_call_output`. Returning a list keeps the
    one-to-many cases honest at the call site.
    """
    if message.role == "tool":
        return [
            {
                "type": "function_call_output",
                "call_id": message.tool_call_id,
                "output": message.content or "",
            }
        ]

    items: list[dict[str, Any]] = []
    # The assistant's prior function calls are replayed as `function_call` items so
    # the model sees its own tool use; their results follow as `function_call_output`
    # on the `tool` turns. The text (if any) is a normal message item.
    if message.content is not None:
        items.append({"role": _input_role(message.role), "content": message.content})
    for call in message.tool_calls:
        items.append(
            {
                "type": "function_call",
                "call_id": call.id,
                "name": call.name,
                "arguments": json.dumps(call.arguments),
            }
        )
    return items


def _input_role(role: str) -> str:
    """Map a harness role to the Responses input-message role.

    The Responses API's first-class instruction role is ``developer`` (the
    documented role for application instructions), not ``system`` as in Chat
    Completions. We map ``system`` → ``developer`` so an agent's charter — which
    the harness carries as a system turn — lands in the role Responses expects;
    ``user`` and ``assistant`` pass through unchanged.
    """
    return "developer" if role == "system" else role


def _function_tool_to_wire(tool: ToolSpec) -> dict[str, Any]:
    """Serialize a `ToolSpec` into the Responses *flat* function-tool shape.

    The Responses API flattens the function tool — `name`/`description`/`parameters`
    sit directly under the tool object, unlike Chat Completions' nested `function`
    key. This is the one wire difference a tool author would otherwise trip on.
    """
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
    }


def _builtin_to_wire(spec: str | Mapping[str, Any]) -> dict[str, Any]:
    """A built-in tool spec as a request tool dict: a type name or a full dict.

    `"web_search"` becomes `{"type": "web_search"}`; a mapping (a built-in that
    carries configuration) is passed through verbatim. This is the registration
    seam — enabling another built-in is adding its name to `builtin_tools`.
    """
    if isinstance(spec, str):
        return {"type": spec}
    return dict(spec)


# --- response: Responses output items → harness Message ----------------------


def _message_from_wire(data: dict[str, Any]) -> Message:
    """Parse a Responses payload into the assistant's `Message`.

    Walks the `output` items by type: `message` items contribute reply text and
    any `url_citation` annotations; `function_call` items become `ToolCall`s for
    the engine to run. Every other item type (`web_search_call`, `reasoning`, and
    future built-ins) is resolved server-side and intentionally skipped here.
    """
    output = data.get("output")
    if not isinstance(output, list):
        raise ProviderError(f"Malformed Responses payload: {data!r}")

    text_parts: list[str] = []
    citations: list[dict[str, Any]] = []
    tool_calls: list[ToolCall] = []

    for item in output:
        kind = item.get("type")
        if kind == "message":
            for part in item.get("content") or []:
                if part.get("type") == "output_text":
                    if part.get("text"):
                        text_parts.append(part["text"])
                    citations.extend(
                        ann
                        for ann in part.get("annotations") or []
                        if ann.get("type") == "url_citation"
                    )
        elif kind == "function_call":
            tool_calls.append(_tool_call_from_wire(item))
        # Any other item type is a server-side built-in result — already resolved
        # by OpenAI, nothing for the engine to do. Skip it.

    text = "".join(text_parts)
    if citations:
        text += _format_citations(citations)
    # Mirror the Chat Completions adapter: a pure tool-call turn carries no text.
    content = text if text else None
    return Message(role="assistant", content=content, tool_calls=tool_calls)


def _tool_call_from_wire(item: dict[str, Any]) -> ToolCall:
    """A Responses `function_call` item as a harness `ToolCall` (arguments parsed)."""
    raw_args = item.get("arguments") or "{}"
    try:
        arguments = json.loads(raw_args)
    except json.JSONDecodeError as exc:
        raise ProviderError(f"Could not decode tool-call arguments {raw_args!r}: {exc}") from exc
    return ToolCall(id=item["call_id"], name=item["name"], arguments=arguments)


def _format_citations(citations: Sequence[dict[str, Any]]) -> str:
    """A `Sources:` footer from `url_citation` annotations, deduplicated by URL.

    Web search grounds the answer in live sources; surfacing them keeps the
    reply honest and checkable. URLs are listed once, in first-seen order, with
    their title when the citation carried one.
    """
    lines: list[str] = []
    seen: set[str] = set()
    for citation in citations:
        url = citation.get("url")
        if not url or url in seen:
            continue
        seen.add(url)
        title = citation.get("title")
        lines.append(f"- {title} — {url}" if title else f"- {url}")
    if not lines:
        return ""
    return "\n\nSources:\n" + "\n".join(lines)
