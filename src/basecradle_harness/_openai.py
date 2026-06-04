"""The one provider adapter v0 ships: any OpenAI-compatible chat endpoint.

The same class talks to OpenAI, OpenRouter, and xAI — they differ only in
`base_url`, `api_key`, and `model`:

    OpenAICompatibleProvider(model="gpt-4o", api_key=...)                       # OpenAI
    OpenAICompatibleProvider(model="x-ai/grok-2", base_url="https://openrouter.ai/api/v1", api_key=...)
    OpenAICompatibleProvider(model="grok-2", base_url="https://api.x.ai/v1", api_key=...)

It implements the `Provider` protocol and owns every wire detail: the
`/chat/completions` POST, bearer auth, the `tool` / `tool_calls` encoding, and
the JSON-string `arguments` that the rest of Harness never has to see.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from typing import Any

import httpx

from basecradle_harness._exceptions import (
    ProviderAPIError,
    ProviderAuthError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
)
from basecradle_harness._messages import Message, ToolCall, ToolSpec

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_TIMEOUT = 60.0


class OpenAICompatibleProvider:
    """A `Provider` backed by any OpenAI-compatible `/chat/completions` endpoint.

    Args:
        model: The model id to request (e.g. ``"gpt-4o"``, ``"grok-2"``).
        api_key: The bearer token. Falls back to ``OPENAI_API_KEY`` when omitted.
        base_url: The API root. Defaults to OpenAI; point it at OpenRouter or
            xAI to use those.
        timeout: Per-request timeout in seconds.
        default_params: Extra body parameters sent on every call (e.g.
            ``temperature=0.2``). ``model``, ``messages``, and ``tools`` always
            take precedence and cannot be overridden here.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        **default_params: Any,
    ) -> None:
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "No API key: pass api_key=... or set the OPENAI_API_KEY environment variable."
            )
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._default_params = default_params
        self._client = httpx.Client(
            headers={"Authorization": f"Bearer {key}"},
            timeout=timeout,
        )

    def chat(self, messages: Sequence[Message], tools: Sequence[ToolSpec] | None = None) -> Message:
        """Run one chat turn and return the model's reply (text and/or tool calls)."""
        payload: dict[str, Any] = dict(self._default_params)
        payload["model"] = self.model
        payload["messages"] = [_message_to_wire(m) for m in messages]
        if tools:
            payload["tools"] = [_tool_to_wire(t) for t in tools]

        try:
            response = self._client.post(f"{self.base_url}/chat/completions", json=payload)
        except httpx.RequestError as exc:
            raise ProviderConnectionError(f"Could not reach the provider: {exc}") from exc

        if response.status_code >= 400:
            _raise_for_status(response)

        return _message_from_wire(response.json())

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> OpenAICompatibleProvider:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _message_to_wire(message: Message) -> dict[str, Any]:
    """Serialize a `Message` into the OpenAI chat shape."""
    if message.role == "tool":
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "content": message.content or "",
        }

    wire: dict[str, Any] = {"role": message.role}
    if message.content is not None:
        wire["content"] = message.content
    if message.tool_calls:
        wire["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
            }
            for call in message.tool_calls
        ]
    # An assistant turn that is purely tool calls still needs an explicit null
    # content key on the wire.
    if message.role == "assistant" and "content" not in wire:
        wire["content"] = None
    return wire


def _tool_to_wire(tool: ToolSpec) -> dict[str, Any]:
    """Serialize a `ToolSpec` into the OpenAI function-tool shape."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def _message_from_wire(data: dict[str, Any]) -> Message:
    """Parse a chat-completions response into the assistant's `Message`."""
    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError(f"Malformed chat completion response: {data!r}") from exc

    tool_calls: list[ToolCall] = []
    for raw in message.get("tool_calls") or []:
        function = raw.get("function", {})
        raw_args = function.get("arguments") or "{}"
        try:
            arguments = json.loads(raw_args)
        except json.JSONDecodeError as exc:
            raise ProviderError(
                f"Could not decode tool-call arguments {raw_args!r}: {exc}"
            ) from exc
        tool_calls.append(ToolCall(id=raw["id"], name=function["name"], arguments=arguments))

    return Message(role="assistant", content=message.get("content"), tool_calls=tool_calls)


def _raise_for_status(response: httpx.Response) -> None:
    """Map an error response onto the typed provider exceptions."""
    status = response.status_code
    body = response.text
    if status in (401, 403):
        raise ProviderAuthError(
            f"Provider rejected the API key (HTTP {status}).", status_code=status, body=body
        )
    if status == 429:
        raise ProviderRateLimitError(
            "Provider rate-limited the request (HTTP 429).",
            status_code=status,
            body=body,
            retry_after=_retry_after(response),
        )
    raise ProviderAPIError(f"Provider returned HTTP {status}.", status_code=status, body=body)


def _retry_after(response: httpx.Response) -> float | None:
    raw = response.headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None
