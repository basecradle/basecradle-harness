"""The native xAI adapter — `grok` over the official ``xai-sdk`` (gRPC), issue #165.

The second `Provider` adapter (after `basecradle_harness._openai.OpenAIProvider`), and the first
that is **not** OpenAI-wire: it reaches grok through xAI's own first-party SDK (`xai-sdk` on PyPI,
``xai-org/xai-sdk-python``), a **gRPC** client — no OpenAI-compatibility shim, no harness-owned
HTTP. Selected by ``AI_SDK=xai-sdk`` (the package name), it is the Grok personas' end-state brain
(issue #165); ``AI_SDK=openai`` pointed at ``api.x.ai`` remains a fully supported alternative cell
(issue #163).

Single native surface
---------------------
The native SDK speaks **one** wire (its gRPC chat service), so this adapter declares a single
`SURFACES` / `DEFAULT_SURFACE` and ``AI_SDK_SURFACE`` is left unset for it — a value other than
the native surface is rejected by `basecradle_harness._basecradle._resolve_surface`.

Brain only — tools are per-persona
----------------------------------
This adapter is the **chat brain** (the `Provider` contract: chat + tool calling). Live Search is
wired here, server-side, when the persona has opted its search built-ins in (issue #168): the
``web_search`` / ``x_search`` built-in names become xAI **Agent Tool** entries
(`xai_sdk.tools.web_search()` / `x_search()`) appended to the request's ``tools`` list, and grok
autonomously runs the search server-side and returns sourced answers with citations. (This replaced
the deprecated native ``SearchParameters`` path — the live gRPC endpoint now rejects it with
``UNIMPLEMENTED: Live search is deprecated`` — issue #171.) grok runs that whole agentic loop
*inside one gRPC turn* and then surfaces **every** tool call it made — the already-executed
server-side ones included — in ``Response.tool_calls``, each tagged by a ``ToolCallType``; the
adapter drops the server-side calls (`_is_client_side`) so they are never re-dispatched to the
harness function registry as bogus ``no tool named`` bounces (issue #183). The grok
**media** tools (`grok_generate_image` / `grok_generate_video`) stay their own per-persona
`PlatformTool`s over httpx (`basecradle_harness._grok`) — independent of the chat SDK, and granted
only by opt-in. Exposing a capability is never granting it to a persona.

Stateless per turn: the full conversation is sent every call and the harness owns history.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from typing import Any

from basecradle_harness._exceptions import (
    ProviderAuthError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
    ProviderResponseError,
)
from basecradle_harness._messages import ImageContent, Message, ToolCall, ToolSpec
from basecradle_harness._openai_wire import format_citations

#: This adapter's single native (gRPC) surface — declared for the SDK-scoped surface contract
#: (issue #163); ``AI_SDK_SURFACE`` is left unset for a single-surface SDK.
SURFACES = ("native",)
DEFAULT_SURFACE = "native"


def require_xai_sdk():
    """Import and return the ``xai_sdk`` package, or raise a clear "no LLM, by design" error.

    The core depends on **no** vendor SDK — an ``AI_SDK=xai-sdk`` agent installs the extra
    (``pip install 'basecradle-harness[xai-sdk]'``). With it absent the harness genuinely cannot
    reach a model, so this fails loud and actionable at provider construction rather than letting a
    bare ``ModuleNotFoundError`` surface from inside a wake.
    """
    try:
        import xai_sdk  # noqa: PLC0415 - lazy: the core must import without the vendor SDK
        import xai_sdk.tools  # noqa: PLC0415 - submodule (Agent Tools) is not auto-imported by __init__
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised via monkeypatched import
        raise ProviderError(
            "The 'xai-sdk' SDK is not installed, so the harness has no way to reach a model "
            "(this is by design — the core depends on no vendor SDK). Install the SDK your "
            "agent's AI_SDK names: pip install 'basecradle-harness[xai-sdk]'."
        ) from exc
    return xai_sdk


class XaiSdkProvider:
    """A `Provider` backed by the official ``xai-sdk`` (gRPC) — grok, natively.

    Satisfies the `Provider` protocol — the engine cannot tell it from any other adapter — but
    every model call goes through the ``xai_sdk`` package, no harness-owned transport.

    Args:
        model: The grok model id (e.g. ``"grok-4.3"``).
        api_key: The xAI bearer token. Falls back to ``AI_API_KEY`` when omitted.
        api_host: The gRPC host. Defaults to the SDK's own (``api.x.ai``).
        timeout: Per-request timeout in seconds (passed to the SDK client).
        builtin_tools: The server-side built-ins a persona has opted in — ``"web_search"`` /
            ``"x_search"`` (issue #168). They are translated to xAI **Agent Tool** entries
            (`xai_sdk.tools`) appended to the request's ``tools`` list so grok runs the search
            itself; a name that maps to no Agent Tool is ignored.
        client: An already-built ``xai_sdk.Client`` (or compatible). The seam tests inject a fake
            through, so the gRPC client is never constructed — and built when omitted.
        default_params: Extra keyword parameters passed to ``chat.create`` on every call (e.g.
            ``temperature=0.2``). ``model``, ``messages``, ``tools`` always take precedence.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        api_host: str | None = None,
        timeout: float | None = None,
        builtin_tools: Sequence[str] = (),
        client: Any | None = None,
        **default_params: Any,
    ) -> None:
        self.model = model
        self._builtin_tools = list(builtin_tools)
        self._default_params = default_params
        self._xai = require_xai_sdk()
        if client is not None:
            self._client = client
        else:
            key = api_key or os.environ.get("AI_API_KEY")
            if not key:
                raise ValueError(
                    "No API key: pass api_key=... or set the AI_API_KEY environment variable."
                )
            kwargs: dict[str, Any] = {"api_key": key}
            if api_host:
                kwargs["api_host"] = api_host
            if timeout is not None:
                kwargs["timeout"] = timeout
            self._client = self._xai.Client(**kwargs)

    def chat(self, messages: Sequence[Message], tools: Sequence[ToolSpec] | None = None) -> Message:
        """Run one model turn through the native SDK and return the assistant's reply."""
        chat_mod = self._xai.chat
        payload: dict[str, Any] = dict(self._default_params)
        payload["model"] = self.model
        payload["messages"] = [self._to_wire(m, chat_mod) for m in messages]
        # Function tools and the opted-in server-side built-ins (search, code execution) share
        # one ``tools`` list: all are native ``chat_pb2.Tool`` protos (issue #171 — Agent Tools).
        wire_tools = [chat_mod.tool(t.name, t.description, t.parameters) for t in tools or ()]
        wire_tools.extend(self._agent_tools())
        if wire_tools:
            payload["tools"] = wire_tools
        with self._mapped_errors():
            conversation = self._client.chat.create(**payload)
            response = conversation.sample()
        return self._from_wire(response)

    # --- wire translation (harness <-> xai_sdk helpers) ----------------------

    def _to_wire(self, message: Message, chat_mod: Any) -> Any:
        """A harness `Message` as an ``xai_sdk`` chat message (a ``chat_pb2.Message``)."""
        role = message.role
        if role == "system":
            return chat_mod.system(message.content or "")
        if role == "developer":
            return chat_mod.developer(message.content or "")
        if role == "tool":
            return chat_mod.tool_result(message.content or "", message.tool_call_id)
        if role == "assistant":
            wire = chat_mod.assistant(message.content or "")
            for call in message.tool_calls:
                wire.tool_calls.append(
                    chat_mod.chat_pb2.ToolCall(
                        id=call.id,
                        function=chat_mod.chat_pb2.FunctionCall(
                            name=call.name, arguments=json.dumps(call.arguments)
                        ),
                    )
                )
            return wire
        # user — text plus any images the engine injected for vision
        if message.images:
            parts = []
            if message.content:
                parts.append(chat_mod.text(message.content))
            parts.extend(self._image_part(img, chat_mod) for img in message.images)
            return chat_mod.user(*parts)
        return chat_mod.user(message.content or "")

    @staticmethod
    def _image_part(image: ImageContent, chat_mod: Any) -> Any:
        """An `ImageContent` as an ``xai_sdk`` image content part."""
        return chat_mod.image(image.url)

    def _from_wire(self, response: Any) -> Message:
        """The SDK's ``Response`` as a harness assistant `Message` (text + tool calls + sources)."""
        tool_calls = [
            ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=json.loads(tc.function.arguments) if tc.function.arguments else {},
            )
            for tc in response.tool_calls
            if self._is_client_side(tc)
        ]
        content = response.content or None
        # Live-Search citations are plain URL strings (xai_sdk Response.citations); footer them
        # through the shared formatter so a grounded grok reply reads like an OpenAI one.
        citations = [{"url": url} for url in getattr(response, "citations", ()) or ()]
        footer = format_citations(citations)
        if footer:
            content = f"{content}\n\n{footer}" if content else footer
        return Message.assistant(content=content, tool_calls=tool_calls)

    def _is_client_side(self, tool_call: Any) -> bool:
        """True unless xAI already ran this tool call **server-side** (issue #183).

        grok runs its whole agentic loop — Live Search (``web_search`` / ``x_search``, with the
        latter's internal X sub-operations), ``code_execution``, and the rest — inside the single
        gRPC turn ``sample()`` makes, then surfaces **every** tool call it made in
        ``Response.tool_calls``, each tagged with a ``ToolCallType``: the already-executed
        server-side ones *and* any genuine client-side function call. Those server-side calls are
        not the harness's to run — re-dispatching one to the function `ToolRegistry` bounces an
        ``Error: no tool named '<x>'`` (the search built-ins read as non-functional, then the model
        confabulates a result). So only a **client-side** call is surfaced; the server-side ones
        are dropped, their results already folded into ``Response.content`` + ``citations``.

        Kept: ``CLIENT_SIDE_TOOL`` (what the SDK tags a real client function call) and the
        unset/``INVALID`` default — the latter both for the offline fakes (which carry no ``type``)
        and as a belt-and-suspenders for an untyped live call. A genuine client call therefore
        always survives. Dropped: every explicit server-side type, named or not, so a server-side
        type xAI adds later is handled the same way without a code change.
        """
        types = self._xai.chat.chat_pb2.ToolCallType
        keep = {types.TOOL_CALL_TYPE_INVALID, types.TOOL_CALL_TYPE_CLIENT_SIDE_TOOL}
        return getattr(tool_call, "type", types.TOOL_CALL_TYPE_INVALID) in keep

    def _agent_tools(self) -> list[Any]:
        """The opted-in server-side built-ins as xAI **Agent Tool** entries (`chat_pb2.Tool`).

        ``web_search`` → `xai_sdk.tools.web_search()`, ``x_search`` → `xai_sdk.tools.x_search()`
        (`x_search` is the single, unified 𝕏 tool — posts, users, and threads), and
        ``code_execution`` → `xai_sdk.tools.code_execution()` (grok writes and runs Python in
        xAI's sandbox — compute only; see the file-I/O note below). grok runs each tool
        server-side and returns the result; the harness never executes it. With nothing opted
        in, returns ``[]`` so the request carries no built-in.

        This is the issue #171 fix: the native ``SearchParameters`` path it replaced is deprecated
        and now rejected by the live gRPC endpoint (``UNIMPLEMENTED: Live search is deprecated``).

        File-I/O asymmetry (issue #172): xAI's ``code_execution`` tool takes no parameters and
        its proto carries no file-input binding — there is **no** input-file mechanism the way
        OpenAI's Code Interpreter container has ``file_ids``. (xAI's *response* proto does carry
        an ``output_files`` field, but whether ``code_execution`` populates it is unverified
        against the live endpoint and is the capital's to confirm on Eddie.) So the Asset bridge
        — `_code.py`, the input/output file round-trip — is **OpenAI-only**; on xAI grok can
        compute but not exchange files with the BaseCradle Asset system. Documented gap, not a
        faked parity.
        """
        tools_mod = self._xai.tools
        builders = {
            "web_search": tools_mod.web_search,
            "x_search": tools_mod.x_search,
            "code_execution": tools_mod.code_execution,
        }
        return [builders[name]() for name in self._builtin_tools if name in builders]

    # --- gRPC errors -> the harness provider error hierarchy ------------------

    def _mapped_errors(self):
        return _grpc_error_context()

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> XaiSdkProvider:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class _grpc_error_context:
    """Map ``xai_sdk``'s gRPC errors onto the harness's typed provider-error hierarchy.

    The engine and tools catch `ProviderConnectionError` / `ProviderError` (and the auth /
    rate-limit subclasses), so gRPC's ``RpcError`` status codes are normalized here, once — an
    auth failure to `ProviderAuthError`, a rate limit to `ProviderRateLimitError`, an unreachable
    endpoint to `ProviderConnectionError`, anything else to a `ProviderError`.
    """

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is None:
            return False
        try:
            import grpc  # noqa: PLC0415 - lazy: only needed to classify, only on the error path
        except ModuleNotFoundError:  # pragma: no cover - grpc ships with xai-sdk
            return False
        if not isinstance(exc, grpc.RpcError):
            return False
        code = exc.code() if callable(getattr(exc, "code", None)) else None
        detail = exc.details() if callable(getattr(exc, "details", None)) else str(exc)
        message = f"xAI gRPC error ({getattr(code, 'name', code)}): {detail}"
        if code == grpc.StatusCode.UNAUTHENTICATED:
            raise ProviderAuthError(message, status_code=401) from exc
        if code == grpc.StatusCode.RESOURCE_EXHAUSTED:
            raise ProviderRateLimitError(message, status_code=429) from exc
        if code in (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED):
            raise ProviderConnectionError(message) from exc
        if code in (grpc.StatusCode.INTERNAL, grpc.StatusCode.DATA_LOSS):
            # A broken/undecodable response payload — gRPC's analogue of the truncated-JSON class
            # (issue #259): the call reached the endpoint and came back corrupt. Map it to the
            # retryable `ProviderResponseError` so the engine re-requests it, the same capability
            # class as the OpenAI/OpenRouter parse failures — a dropped wake, not a config bug.
            raise ProviderResponseError(message) from exc
        raise ProviderError(message) from exc
