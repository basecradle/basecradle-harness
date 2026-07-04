"""The native OpenRouter adapter — models over OpenRouter's first-party Python SDK.

The third `Provider` adapter (after `basecradle_harness._openai.OpenAIProvider` and
`basecradle_harness._xai_sdk.XaiSdkProvider`): it reaches a model through OpenRouter's own
official SDK (`openrouter` on PyPI, ``OpenRouterTeam/python-sdk``) — a Speakeasy-generated,
httpx-backed client — no OpenAI-compatibility shim, no harness-owned HTTP. Selected by
``AI_SDK=openrouter`` (the package name), it is the brain for the @glm-5.2 peer (``z-ai/glm-5.2``)
and any other OpenRouter-hosted model.

OpenRouter speaks the OpenAI **chat** wire, so this adapter *reuses* the shared, transport-free
`basecradle_harness._openai_wire` translation exactly as the chat surface of the openai adapter
does — it is only *SDK plumbing*: build the request dict, call the SDK, parse
``response.model_dump()`` back. There is a second, fully-supported way to reach OpenRouter — the
``openai`` SDK pointed at ``openrouter.ai`` over its chat surface (see
`basecradle_harness._basecradle`) — so this native adapter and the openai-at-OpenRouter cell are
a permanent matrix, not either/or.

Single chat surface
-------------------
OpenRouter's Responses API is beta upstream, so this adapter declares a single ``chat`` surface;
``AI_SDK_SURFACE`` is left unset for it (and the openrouter-via-``openai``-SDK cell is likewise
gated chat-only in the config layer, with a clear error naming the fix).

Typed ``chat.send`` — the model-params caveat
--------------------------------------------
Unlike the ``openai`` SDK, OpenRouter's ``chat.send`` is fully **typed and keyword-only with no
``**kwargs``** — an unknown keyword raises ``TypeError`` at call time, and there is **no**
``extra_body`` escape hatch. So an operator's ``model_params.json`` reaches this SDK only through
the keys ``chat.send`` actually names (``temperature``, ``max_tokens``, ``reasoning``,
``reasoning_effort``, ``top_p``, …); a key it does not name is turned, in the error mapper, into
an actionable `ProviderError` naming ``model_params.json`` rather than a bare ``TypeError`` from
inside a wake.

Stateless per turn, like the wire it speaks: the full conversation is sent every call and the
harness owns history — this adapter never sets ``stream`` (it is non-streaming by contract).
"""

from __future__ import annotations

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
from basecradle_harness._messages import Message, ToolSpec
from basecradle_harness._openai_wire import (
    chat_message_to_wire,
    chat_tool_to_wire,
    message_from_chat,
)

#: OpenRouter's API root — supplied as the SDK ``server_url`` (its own default is the same host,
#: but the harness passes it explicitly so the config layer's ``AI_BASE_URL`` override flows here).
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TIMEOUT = 60.0
#: This adapter's single surface — declared for the SDK-scoped surface contract (issue #163).
#: OpenRouter's Responses API is beta upstream, so only the OpenAI-compatible ``chat`` wire ships;
#: ``AI_SDK_SURFACE`` is left unset for it.
SURFACES = ("chat",)
DEFAULT_SURFACE = "chat"


def require_openrouter_sdk():
    """Import and return the ``openrouter`` package, or raise a clear "no LLM, by design" error.

    The core has **no** vendor-SDK dependency — an ``AI_SDK=openrouter`` agent installs only the
    extra (``pip install 'basecradle-harness[openrouter]'``). With it absent the harness genuinely
    cannot reach a model, so this fails loud and actionable at provider construction rather than
    letting a bare ``ModuleNotFoundError`` surface from inside a wake.
    """
    try:
        import openrouter  # noqa: PLC0415 - lazy: the core must import without the vendor SDK
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised via monkeypatched import
        raise ProviderError(
            "The 'openrouter' SDK is not installed, so the harness has no way to reach a model "
            "(this is by design — the core depends on no vendor SDK). Install the SDK your "
            "agent's AI_SDK names: pip install 'basecradle-harness[openrouter]'."
        ) from exc
    return openrouter


class OpenRouterProvider:
    """A `Provider` backed by OpenRouter's official ``openrouter`` SDK (chat wire).

    Satisfies the `Provider` protocol — the engine cannot tell it from any other adapter — but
    every model call goes through the ``openrouter`` package, no harness-owned HTTP. The wire
    translation is the shared `basecradle_harness._openai_wire` (OpenRouter speaks the OpenAI chat
    wire), so this class is just SDK plumbing.

    Args:
        model: The OpenRouter model id, vendor-prefixed (e.g. ``"z-ai/glm-5.2"``).
        api_key: The OpenRouter bearer token. Falls back to ``AI_API_KEY`` when omitted.
        base_url: The API root, passed to the SDK as ``server_url``. Defaults to OpenRouter's own.
        timeout: Per-request timeout in seconds (passed to the SDK as ``timeout_ms``).
        client: An already-built ``openrouter.OpenRouter`` (or compatible). The seam tests inject a
            client through, so the httpx client need not be constructed; built when omitted.
        default_params: Extra keyword parameters passed to ``chat.send`` on every call, sourced
            from the operator's ``model_params.json`` (e.g. ``temperature=0.2``,
            ``reasoning={"effort": "high"}``). ``model``, ``messages``, ``tools`` always take
            precedence. Because ``chat.send`` is typed with no ``**kwargs``, a key it does not name
            raises a ``TypeError`` mapped to an actionable `ProviderError` (see `_ErrorMapper`).
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        client: Any | None = None,
        **default_params: Any,
    ) -> None:
        self.model = model
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self._default_params = default_params
        self._openrouter = require_openrouter_sdk()
        if client is not None:
            self._client = client
        else:
            key = api_key or os.environ.get("AI_API_KEY")
            if not key:
                raise ValueError(
                    "No API key: pass api_key=... or set the AI_API_KEY environment variable."
                )
            self._client = self._openrouter.OpenRouter(
                api_key=key,
                server_url=base_url or None,
                timeout_ms=int(timeout * 1000),
                # Disable the SDK's default 5xx retry. Its Speakeasy default backs off up to
                # ~1 hour (BackoffStrategy max_elapsed 3_600_000ms) on a persistent 5xx, which
                # would hang a wake far past ``timeout`` (that is per-attempt, not total). The
                # harness fails a wake fast and lets the router re-wake on the next event —
                # matching the other adapters' bounded behavior (openai ``max_retries``, the
                # xAI gRPC deadline). This also makes production match what the tests exercise.
                retry_config=None,
            )

    def chat(self, messages: Sequence[Message], tools: Sequence[ToolSpec] | None = None) -> Message:
        """Run one model turn through the SDK and return the assistant's reply."""
        payload: dict[str, Any] = dict(self._default_params)
        payload["model"] = self.model
        payload["messages"] = [chat_message_to_wire(m) for m in messages]
        if tools:
            payload["tools"] = [chat_tool_to_wire(t) for t in tools]
        # Never set ``stream``: this adapter is non-streaming by contract, and a truthy ``stream``
        # would change ``chat.send``'s return type from ``ChatResult`` to an event stream.
        with self._mapped_errors():
            response = self._client.chat.send(**payload)
        return message_from_chat(response.model_dump())

    # --- SDK exceptions → the harness provider error hierarchy ----------------

    def _mapped_errors(self):
        """A context manager mapping ``openrouter`` SDK exceptions onto the harness typed errors.

        The engine and tools catch `ProviderConnectionError` / `ProviderAPIError` and its
        subclasses, so the SDK's own exception zoo is normalized here, once, into that contract.
        """
        return _ErrorMapper(self._openrouter)

    def close(self) -> None:
        """Best-effort close of the SDK's underlying httpx client.

        The ``openrouter`` SDK exposes no public ``close()`` — a ``weakref.finalize`` closes the
        httpx clients when the SDK is collected. We still close the sync client explicitly when we
        built it (it lives at ``sdk.sdk_configuration.client``) so a long-lived process reclaims
        the socket promptly; the finalizer remains the backstop. Guarded so an injected fake
        client (a test double without the attribute) never raises here.
        """
        config = getattr(self._client, "sdk_configuration", None)
        http_client = getattr(config, "client", None)
        close = getattr(http_client, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> OpenRouterProvider:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class _ErrorMapper:
    """Translate ``openrouter`` SDK exceptions into the harness `ProviderError` hierarchy.

    The SDK raises `openrouter.errors.OpenRouterError` (its base, carrying ``.status_code``,
    ``.body``, ``.headers``) for an HTTP error response, `NoResponseError` when nothing came back,
    and lets raw ``httpx`` transport exceptions propagate. An **unexpected-keyword-argument**
    `TypeError` from the ``chat.send(**payload)`` splat means an operator put a key in
    ``model_params.json`` that the typed, no-``**kwargs`` ``chat.send`` does not accept — turned
    here into an actionable error naming the file, not a bare ``TypeError`` from inside a wake. Any
    *other* ``TypeError`` (a genuine bug in message/tool marshalling deep in the SDK) is left to
    propagate unchanged, so it is never misattributed to a possibly-empty model-params file.
    """

    def __init__(self, openrouter) -> None:
        self._openrouter = openrouter

    def __enter__(self) -> _ErrorMapper:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is None:
            return False
        errors = self._openrouter.errors
        if isinstance(exc, errors.OpenRouterError):
            raise _from_status_error(exc) from exc
        if isinstance(exc, errors.NoResponseError):
            raise ProviderConnectionError(f"Could not reach OpenRouter: {exc}") from exc
        if isinstance(exc, httpx.RequestError):
            # DNS/TCP/TLS/timeout — nothing reached the model.
            raise ProviderConnectionError(f"Could not reach OpenRouter: {exc}") from exc
        if isinstance(exc, TypeError) and "unexpected keyword argument" in str(exc):
            # ``chat.send`` is typed with no ``**kwargs``; an unknown key came from model_params.
            # Only the unexpected-keyword-argument TypeError is ours to reframe — any other
            # TypeError (an SDK marshalling bug) propagates so it is not blamed on model_params.
            raise ProviderError(
                f"A key in model_params.json was not accepted by the openrouter SDK's "
                f"chat.send: {exc}. Its parameters are a typed set (temperature, max_tokens, "
                "reasoning, reasoning_effort, top_p, …); remove the offending key."
            ) from exc
        return False  # not an SDK/transport error — let it propagate unchanged


def _from_status_error(exc) -> ProviderError:
    """An ``openrouter.errors.OpenRouterError`` mapped to the right typed `ProviderError`.

    Most subclasses carry a real HTTP error status (401/403/429/5xx) → the matching
    `ProviderAPIError`. But `ResponseValidationError` also subclasses `OpenRouterError` while
    carrying the *success* status of the response whose body failed to parse (e.g. 200) — a
    non-HTTP-error status. Rather than stamp a `ProviderAPIError` with a misleading `status_code`
    like 200 (or a fabricated 0 for a status-less error), a status below 400 is surfaced as a plain
    `ProviderError`: it *is* a provider failure, just not an HTTP-status one.
    """
    status = getattr(exc, "status_code", None) or 0
    body = getattr(exc, "body", "") or ""
    message = getattr(exc, "message", None) or str(exc)
    if status < 400:
        return ProviderError(message)
    if status in (401, 403):
        return ProviderAuthError(
            f"OpenRouter rejected the API key (HTTP {status}).", status_code=status, body=body
        )
    if status == 429:
        return ProviderRateLimitError(
            "OpenRouter rate-limited the request (HTTP 429).",
            status_code=status,
            body=body,
            retry_after=_retry_after(exc),
        )
    # Carry OpenRouter's own message + body so a caller can relay the true cause.
    return ProviderAPIError(message, status_code=status, body=body)


def _retry_after(exc) -> float | None:
    """The ``Retry-After`` seconds hinted on a 429, from the SDK error's ``.headers``, if numeric."""
    headers = getattr(exc, "headers", None)
    if headers is None:
        return None
    raw = headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None
