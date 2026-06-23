"""The xAI interim adapter — hand-rolled httpx over the Responses wire, on death row.

**This is the one provider path the harness still drives with its own HTTP, and it is
temporary.** The architecture is *harness ↔ LLM only through a vendor SDK*; Milestone 1
delivered that for the ``openai`` SDK (`basecradle_harness._openai`). xAI's native ``xai-sdk``
adapter is a later milestone, so until then the ``xai`` profile keeps reaching grok over this
hand-rolled httpx — **left as-is on purpose** (a founder decision), not routed through the
``openai`` SDK (the lowest-common-denominator path being eliminated) and not disabled (it must
not break the live xAI personas).

xAI's API speaks the **Responses wire** (it deprecated Chat Completions ``search_parameters``
in favor of server-side search tools on Responses), so the "OpenAI" heritage here is the wire
format, not the vendor: this adapter POSTs ``/responses`` at ``api.x.ai`` and lets grok run its
Live-Search built-ins (``web_search`` / ``x_search``) alongside the agent's function tools. The
wire translation is the shared, transport-free `basecradle_harness._openai_wire` — the same
functions the SDK adapter uses — so this file is now just the httpx envelope around them.

Stateless per turn: the full conversation is sent every call and the harness owns history.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

import httpx

from basecradle_harness._exceptions import ProviderConnectionError
from basecradle_harness._http import raise_for_status
from basecradle_harness._messages import Message, ToolSpec
from basecradle_harness._openai_wire import (
    builtin_to_responses,
    function_tool_to_responses,
    message_from_responses,
    message_to_input,
)

#: xAI's Responses root — this adapter's default endpoint.
DEFAULT_BASE_URL = "https://api.x.ai/v1"
DEFAULT_TIMEOUT = 60.0


class OpenAIResponsesProvider:
    """A `Provider` that POSTs the Responses wire over httpx — the xAI interim path.

    Satisfies the same `Provider` protocol as the SDK adapter, but talks to ``/responses``
    directly so xAI can run its Live-Search built-ins (``web_search`` / ``x_search``) alongside
    the agent's custom function tools. **Death row:** the ``xai-sdk`` adapter replaces it.

    Args:
        model: The model id (e.g. ``"grok-4.3"``).
        api_key: The bearer token. Falls back to ``AI_API_KEY`` when omitted.
        base_url: The API root. Defaults to ``api.x.ai``; ``AI_BASE_URL`` overrides upstream.
        timeout: Per-request timeout in seconds.
        builtin_tools: The server-side built-ins to enable, as type names (``"web_search"``,
            ``"x_search"``) or full tool dicts. Merged with the function tools each turn.
        default_params: Extra body parameters sent on every call. ``model``, ``input``, and
            ``tools`` always take precedence.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        builtin_tools: Sequence[str | Mapping[str, Any]] = (),
        **default_params: Any,
    ) -> None:
        key = api_key or os.environ.get("AI_API_KEY")
        if not key:
            raise ValueError(
                "No API key: pass api_key=... or set the AI_API_KEY environment variable."
            )
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._builtin_tools = [builtin_to_responses(spec) for spec in builtin_tools]
        self._default_params = default_params
        self._client = httpx.Client(
            headers={"Authorization": f"Bearer {key}"},
            timeout=timeout,
        )

    def chat(self, messages: Sequence[Message], tools: Sequence[ToolSpec] | None = None) -> Message:
        """Run one Responses turn and return the model's reply (text and/or tool calls)."""
        payload: dict[str, Any] = dict(self._default_params)
        payload["model"] = self.model
        payload["input"] = [item for m in messages for item in message_to_input(m)]
        wire_tools = list(self._builtin_tools)
        if tools:
            wire_tools.extend(function_tool_to_responses(t) for t in tools)
        if wire_tools:
            payload["tools"] = wire_tools

        try:
            response = self._client.post(f"{self.base_url}/responses", json=payload)
        except httpx.RequestError as exc:
            raise ProviderConnectionError(f"Could not reach the provider: {exc}") from exc

        if response.status_code >= 400:
            raise_for_status(response)

        return message_from_responses(response.json())

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> OpenAIResponsesProvider:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
