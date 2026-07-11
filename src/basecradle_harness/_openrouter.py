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

Web search — a server tool
--------------------------
OpenRouter runs web search entirely server-side (its ``openrouter:web_search`` **server tool**,
issue #237): when the opted-in ``web_search`` built-in is active the adapter puts
``{"type": "openrouter:web_search", "parameters": …}`` on the chat ``tools`` array, OpenRouter
searches and feeds the results back, and returns a grounded, cited answer — the harness never
executes anything (the same safe-by-construction shape as the OpenAI/xAI web-search built-ins).
The ``parameters`` come verbatim from the operator's ``search_params.json``. The catch is on the
*response* side: the SDK's typed ``ChatResult`` does not model the ``url_citation`` annotations
the search returns, so they are recovered from the raw body via a response event hook
(`_ResponseCapture`) and footered by the shared `_openai_wire.message_from_chat` — see
`OpenRouterProvider._restore_annotations`.

Stateless per turn, like the wire it speaks: the full conversation is sent every call and the
harness owns history — this adapter never sets ``stream`` (it is non-streaming by contract).
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence
from typing import Any

import httpx

from basecradle_harness._exceptions import (
    ProviderAPIError,
    ProviderAuthError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
    ProviderResponseError,
)
from basecradle_harness._messages import Message, ToolSpec
from basecradle_harness._observability import log_llm_call
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
#: The endpoint vendor this adapter reaches — the native SDK talks only to OpenRouter, so it is a
#: class constant rather than a constructor arg. It rides the per-call log line as
#: ``provider=openrouter``.
PROVIDER = "openrouter"

#: The model-facing name of the web-search built-in — the one server tool that ships (issue #237)
#: and the only one that consumes ``search_params.json``. Named so the config layer can gate the
#: search-params read on it without hardcoding the string.
WEB_SEARCH_BUILTIN = "web_search"

#: Model-facing built-in name → OpenRouter **server-tool** wire ``type``. A server tool rides the
#: chat ``tools`` array as ``{"type": <wire type>}`` and OpenRouter runs it entirely server-side —
#: the harness never executes it. ``web_search`` is the one that ships (issue #237); OpenRouter
#: offers more (``openrouter:datetime``, ``openrouter:bash``, …) that a future built-in slots in
#: here. A built-in name absent from this map is not an OpenRouter server tool and is skipped.
_SERVER_TOOL_TYPES = {WEB_SEARCH_BUILTIN: "openrouter:web_search"}


class _ResponseCapture:
    """An httpx ``response`` event hook that stashes the last response's parsed JSON body.

    Why the harness reads the raw body at all: the ``openrouter`` SDK's typed ``ChatResult`` only
    keeps the fields it models, and it does **not** model web-search ``url_citation`` annotations
    (v0.11.3 has no ``annotations`` field) — so ``response.model_dump()`` has already lost them by
    the time the adapter runs. This hook captures the raw body the SDK *received*, letting the
    adapter graft the citations back (`OpenRouterProvider._restore_annotations`) and footer them
    like every other web-search built-in. The SDK still owns the request/response cycle; this only
    observes its result, so the "reach a model only through the vendor SDK" contract holds.

    Single-threaded by contract — one provider per agent, one wake at a time — so "last response"
    is unambiguous: the hook writes `last` during ``chat.send`` and the adapter reads it on the
    same thread immediately after. A non-JSON or error body just yields ``None`` (no annotations to
    restore); the adapter only consults `last` on the success path.
    """

    def __init__(self) -> None:
        self.last: Any = None

    def __call__(self, response: httpx.Response) -> None:
        # Read (and cache) the body so both this hook and the SDK's own parse can consume it; a
        # streaming/binary/error body that isn't JSON simply leaves `last` as None.
        try:
            response.read()
            self.last = response.json()
        except Exception:  # noqa: BLE001 - observation is best-effort; never disturb the SDK call
            self.last = None


def _server_tool_objects(
    builtins: Sequence[str], web_search_params: Mapping[str, Any] | None
) -> list[dict[str, Any]]:
    """The active built-ins as OpenRouter server-tool wire objects, in order.

    Each known built-in name becomes ``{"type": <wire type>}``; the ``web_search`` tool also
    carries the operator's ``search_params.json`` verbatim as its ``parameters`` block when any is
    set (absent/empty → the bare object, letting OpenRouter's defaults ride). A name that is not a
    known OpenRouter server tool is skipped rather than sent as an unknown tool the endpoint would
    reject — the resolver only feeds names an active plugin claimed, so in practice this is
    ``web_search`` alone today.
    """
    objects: list[dict[str, Any]] = []
    for name in builtins:
        wire_type = _SERVER_TOOL_TYPES.get(name)
        if wire_type is None:
            continue
        obj: dict[str, Any] = {"type": wire_type}
        if name == "web_search" and web_search_params:
            obj["parameters"] = dict(web_search_params)
        objects.append(obj)
    return objects


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
        builtin_tools: The active server-side built-ins to enable, as model-facing names
            (``"web_search"``). Resolved from the active tool plugins and turned into OpenRouter
            server-tool objects (`_server_tool_objects`) that ride the chat ``tools`` array ahead
            of the custom function tools each turn. A name that is not a known OpenRouter server
            tool is skipped. Empty (the default) → no server tool is sent.
        web_search_params: The operator's ``search_params.json``, passed verbatim as the
            ``openrouter:web_search`` tool's ``parameters`` block (engine, result caps, domain
            filters, …). ``None``/empty → the bare tool object, so OpenRouter's own defaults ride.
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
        builtin_tools: Sequence[str] = (),
        web_search_params: Mapping[str, Any] | None = None,
        **default_params: Any,
    ) -> None:
        self.model = model
        self.provider = PROVIDER
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self._default_params = default_params
        # Server-tool objects are config-time constant (they don't vary per turn), so build them
        # once here rather than on every `chat` call.
        self._server_tools = _server_tool_objects(builtin_tools, web_search_params)
        self._openrouter = require_openrouter_sdk()
        if client is not None:
            # An injected client (the seam tests) carries no capture hook: web-search citations,
            # which are recovered from the raw body, are simply not restored on this path — the
            # answer is still correct, just un-footered. Production always builds its own client.
            self._client = client
            self._capture: _ResponseCapture | None = None
        else:
            key = api_key or os.environ.get("AI_API_KEY")
            if not key:
                raise ValueError(
                    "No API key: pass api_key=... or set the AI_API_KEY environment variable."
                )
            # Install the response-capturing httpx client **only when a server tool is active** —
            # its sole purpose is recovering web-search `url_citation` annotations, so a default
            # agent with no web search pays no capture or extra-parse overhead and lets the SDK
            # build its own default client. When present, the SDK still makes and parses every
            # call; the hook only *observes* the response the SDK already fetched, to recover the
            # annotations the SDK's typed ChatResult drops (it has no annotations field; see
            # `_restore_annotations`). This is not harness-owned HTTP: the SDK owns the
            # request/response cycle; the hook reads its result.
            http_client: Any | None = None
            if self._server_tools:
                self._capture = _ResponseCapture()
                http_client = httpx.Client(
                    timeout=timeout, event_hooks={"response": [self._capture]}
                )
            else:
                self._capture = None
            self._client = self._openrouter.OpenRouter(
                api_key=key,
                server_url=base_url or None,
                client=http_client,
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
        # Server-side built-ins (web_search) lead the tools array, then the custom function tools —
        # the same order the OpenAI Responses adapter uses. OpenRouter runs the server tools itself
        # and returns the model's answer already grounded, with url_citation annotations.
        wire_tools = list(self._server_tools)
        if tools:
            wire_tools.extend(chat_tool_to_wire(t) for t in tools)
        if wire_tools:
            payload["tools"] = wire_tools
        # Never set ``stream``: this adapter is non-streaming by contract, and a truthy ``stream``
        # would change ``chat.send``'s return type from ``ChatResult`` to an event stream.
        started = time.monotonic()
        with self._mapped_errors():
            response = self._client.chat.send(**payload)
        data = response.model_dump()
        log_llm_call(
            provider=self.provider,
            model=self.model,
            seconds=time.monotonic() - started,
            usage=data.get("usage"),
        )
        self._restore_annotations(data)
        return message_from_chat(data)

    def _restore_annotations(self, data: dict[str, Any]) -> None:
        """Graft the web-search ``url_citation`` annotations back onto the SDK's model_dump.

        The SDK's typed ``ChatResult`` has no ``annotations`` field, so it silently drops the
        web-search citations from the body before the harness sees them. `message_from_chat`
        footers ``url_citation`` annotations exactly as the Responses surface does — so, keyed by
        choice index, this copies the annotations from the captured *raw* body onto each dumped
        message, lighting up that one shared citation path. No capture (an injected client), no raw
        body, or no annotations → ``data`` is untouched and the reply is un-footered but correct.
        """
        raw = self._capture.last if self._capture is not None else None
        if not isinstance(raw, Mapping):
            return
        raw_choices = raw.get("choices")
        if not isinstance(raw_choices, list):
            return
        for i, choice in enumerate(data.get("choices") or []):
            if i >= len(raw_choices):
                break
            raw_choice = raw_choices[i]
            raw_message = raw_choice.get("message") if isinstance(raw_choice, Mapping) else None
            annotations = (
                raw_message.get("annotations") if isinstance(raw_message, Mapping) else None
            )
            message = choice.get("message") if isinstance(choice, Mapping) else None
            if annotations and isinstance(message, dict):
                message["annotations"] = annotations

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
        if isinstance(exc, errors.ResponseValidationError):
            # A 200 whose body the SDK's typed ChatResult could not parse — the truncated /
            # EOF-mid-JSON class (issue #259). It is a `ResponseValidationError` (an
            # OpenRouterError subclass) but *transient*, so it maps to the retryable
            # `ProviderResponseError`, not a permanent one; the engine re-requests it. Checked
            # before the generic OpenRouterError branch below, which it subclasses.
            raise ProviderResponseError(str(exc)) from exc
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
    `ProviderAPIError`. The `ResponseValidationError` case (a 200 whose body failed to parse) is
    intercepted earlier by `_ErrorMapper` and mapped to the retryable `ProviderResponseError`, so
    it never reaches here. Any *other* status-less or sub-400 OpenRouterError is still surfaced as
    a plain `ProviderError` rather than a `ProviderAPIError` stamped with a misleading status like
    200 (or a fabricated 0): it *is* a provider failure, just not an HTTP-status one.
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
