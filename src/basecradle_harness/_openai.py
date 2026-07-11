"""The ``openai`` vendor-SDK adapter — the one provider adapter v0 ships.

The harness reaches an LLM **only through a vendor's official SDK**, never hand-rolled HTTP.
This is that adapter for ``AI_SDK=openai``: a thin wrapper over the real ``openai`` package
that satisfies the `Provider` seam. It drives @jt's whole model stack — the conversation loop,
the server-side ``web_search`` built-in, function/tool calling, and vision (image input) —
through ``client.responses`` / ``client.chat.completions``, so the harness ships zero of its
own code to hit a model endpoint.

Two surfaces, one adapter
-------------------------
``surface`` is an **internal option** of this adapter, not a top-level config axis:

- ``"responses"`` (the default, @jt's surface) → ``client.responses.create``. The only path
  that runs server-side built-ins (``web_search``) and the path that sees images.
- ``"chat"`` → ``client.chat.completions.create``. The portable Chat Completions surface, for
  an OpenAI-compatible endpoint (a later milestone's OpenRouter) that lacks Responses.

The wire translation for both is the shared, transport-free `basecradle_harness._openai_wire`
— so this class is just *SDK plumbing*: build the request dict, call the SDK, parse
``response.model_dump()`` back. This one adapter also serves the **xAI profile**, by pointing
the same ``openai`` client at ``api.x.ai`` (issue #163) — see `basecradle_harness._basecradle`. The
``openai`` package is an **optional extra** (``pip install basecradle-harness[openai]``); with
it absent, constructing this adapter raises a clear "no LLM, by design" error rather than a
bare ``ModuleNotFoundError`` deep in a wake.

Stateless per turn, like the wire it speaks: the full conversation is sent every call and the
harness owns history, so Responses' server-side state (``previous_response_id``) is unused.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

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
    builtin_to_responses,
    chat_message_to_wire,
    chat_tool_to_wire,
    function_tool_to_responses,
    message_from_chat,
    message_from_responses,
    message_to_input,
)

#: OpenAI's default API root — what the SDK targets when no ``base_url`` is given.
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_TIMEOUT = 60.0
#: This adapter's surfaces (see the module docstring), as an **SDK-scoped** declaration: the
#: harness reads ``AI_SDK_SURFACE`` against the *active SDK adapter's* ``SURFACES`` (omitted →
#: ``DEFAULT_SURFACE``; provided-but-unlisted → hard fail). ``responses`` is @jt's default.
SURFACES = ("responses", "chat")
#: The surface used when ``AI_SDK_SURFACE`` is unset — this adapter's default wire surface.
DEFAULT_SURFACE = "responses"


def require_openai_sdk():
    """Import and return the ``openai`` package, or raise a clear "no LLM, by design" error.

    The core has **no** vendor-SDK dependency — an agent installs only the extra its ``AI_SDK``
    names (`pip install basecradle-harness[openai]`). When that extra is absent the harness
    genuinely cannot reach a model, so this fails loud and actionable at provider construction
    rather than letting a bare ``ModuleNotFoundError`` surface from inside a wake.
    """
    try:
        import openai  # noqa: PLC0415 - lazy: the core must import without the vendor SDK
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised via monkeypatched import
        raise ProviderError(
            "The 'openai' SDK is not installed, so the harness has no way to reach a model "
            "(this is by design — the core depends on no vendor SDK). Install the SDK your "
            "agent's AI_SDK names: pip install 'basecradle-harness[openai]'."
        ) from exc
    return openai


class OpenAIProvider:
    """A `Provider` backed by the official ``openai`` SDK (Responses or Chat Completions).

    Satisfies the `Provider` protocol — the engine cannot tell it from any other adapter — but
    every model call goes through the ``openai`` package, no harness-owned HTTP.

    Args:
        model: The model id (e.g. ``"gpt-5.4-mini"``).
        api_key: The OpenAI bearer token. Falls back to ``AI_API_KEY`` when omitted.
        base_url: The API root. Defaults to OpenAI; set it for an OpenAI-compatible endpoint.
        provider: The endpoint's vendor (``AI_PROVIDER``) — a **label**, not wiring: this one
            adapter serves OpenAI, xAI, and OpenRouter alike, and only `_provider_from_config`
            knows which endpoint it aimed the client at. It rides the per-call log line so a
            grok-through-the-openai-SDK wake reads ``provider=xai``, not ``provider=openai``.
        surface: ``"responses"`` (default) or ``"chat"`` — this adapter's internal wire
            surface (see the module docstring). Server-side built-ins and vision require
            ``"responses"``.
        timeout: Per-request timeout in seconds.
        max_retries: How many times the SDK retries a transient failure. Defaults to the SDK's
            own resilience (2); set 0 for a single-shot call.
        builtin_tools: The server-side built-ins to enable on the Responses surface, as type
            names (``"web_search"``) or full tool dicts. Resolved from the active tool plugins
            and merged with the custom function tools each turn. Ignored on the chat surface.
        extra_body: Non-standard top-level body fields forwarded as-is on **every** call (both
            surfaces) through the SDK's own ``extra_body`` passthrough. The adapter stays
            vendor-neutral — this is the seam for a provider-specific field the typed SDK params
            don't cover, e.g. xAI's ``search_parameters`` Live-Search object when the ``openai``
            SDK is pointed at ``api.x.ai`` (see `basecradle_harness._basecradle`).
        code_container: An optional callback supplying the ``container`` config for the
            ``code_interpreter`` built-in, evaluated **per turn** (the container handle changes
            as the Asset bridge stages files / pins a session — see `_code.py`). Returns a
            container id string, a container dict, or ``None``. When absent (or it returns
            ``None``) the built-in falls back to ``{"type": "auto"}``. The adapter stays
            BaseCradle-agnostic — it just asks "what container?"; the bridge answers.
        default_params: Extra body parameters sent on every call (e.g. ``temperature=0.2``).
            ``model``, the input/messages, and ``tools`` always take precedence.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        provider: str = "openai",
        surface: str = DEFAULT_SURFACE,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = 2,
        builtin_tools: Sequence[str | Mapping[str, Any]] = (),
        extra_body: Mapping[str, Any] | None = None,
        code_container: Callable[[], dict[str, Any] | str | None] | None = None,
        **default_params: Any,
    ) -> None:
        if surface not in SURFACES:
            raise ValueError(f"Unknown surface {surface!r}; expected one of {SURFACES}.")
        key = api_key or os.environ.get("AI_API_KEY")
        if not key:
            raise ValueError(
                "No API key: pass api_key=... or set the AI_API_KEY environment variable."
            )
        openai = require_openai_sdk()
        self.model = model
        self.provider = provider
        self.surface = surface
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self._builtin_tools = [builtin_to_responses(spec) for spec in builtin_tools]
        self._code_container = code_container
        self._extra_body = dict(extra_body) if extra_body else None
        self._default_params = default_params
        self._openai = openai
        self._client = openai.OpenAI(
            api_key=key,
            base_url=base_url or None,
            timeout=timeout,
            max_retries=max_retries,
        )

    def chat(self, messages: Sequence[Message], tools: Sequence[ToolSpec] | None = None) -> Message:
        """Run one model turn through the SDK and return the assistant's reply."""
        if self.surface == "responses":
            return self._responses_turn(messages, tools)
        return self._chat_turn(messages, tools)

    # --- the two surfaces ----------------------------------------------------

    def _responses_turn(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec] | None
    ) -> Message:
        payload: dict[str, Any] = dict(self._default_params)
        payload["model"] = self.model
        payload["input"] = [item for m in messages for item in message_to_input(m)]
        wire_tools = [self._with_code_container(spec) for spec in self._builtin_tools]
        if tools:
            wire_tools.extend(function_tool_to_responses(t) for t in tools)
        if wire_tools:
            payload["tools"] = wire_tools
        if self._extra_body:
            payload["extra_body"] = dict(self._extra_body)
        started = time.monotonic()
        with self._mapped_errors():
            response = self._client.responses.create(**payload)
        data = response.model_dump()
        self._log_call(started, data)
        return message_from_responses(data)

    def _with_code_container(self, spec: dict[str, Any]) -> dict[str, Any]:
        """Inject the live ``container`` into the ``code_interpreter`` built-in, per turn.

        The Code Interpreter built-in needs a container (auto-created, or a pinned session id
        once the Asset bridge knows one), and that handle changes during a wake — so it cannot
        be baked in at construction. Every other built-in passes through untouched. With no
        `code_container` callback (or it returns ``None``) the built-in falls back to an
        auto-created container, exactly as the bare built-in would.
        """
        if spec.get("type") != "code_interpreter":
            return spec
        container = self._code_container() if self._code_container is not None else None
        return {**spec, "container": container if container is not None else {"type": "auto"}}

    def _chat_turn(self, messages: Sequence[Message], tools: Sequence[ToolSpec] | None) -> Message:
        payload: dict[str, Any] = dict(self._default_params)
        payload["model"] = self.model
        payload["messages"] = [chat_message_to_wire(m) for m in messages]
        if tools:
            payload["tools"] = [chat_tool_to_wire(t) for t in tools]
        if self._extra_body:
            payload["extra_body"] = dict(self._extra_body)
        started = time.monotonic()
        with self._mapped_errors():
            response = self._client.chat.completions.create(**payload)
        data = response.model_dump()
        self._log_call(started, data)
        return message_from_chat(data)

    def _log_call(self, started: float, data: Mapping[str, Any]) -> None:
        """The one INFO line this call earns: provider, model, duration, tokens.

        Both surfaces report usage, under different names (Responses ``input_tokens`` vs Chat
        ``prompt_tokens``) — `log_llm_call` reads either, so this one call site serves both. Only
        a call that *returned* is logged; a call that raised is the error path's story to tell
        (the engine logs the retry/give-up), and timing a failure as if it were a completion
        would be a lie.
        """
        log_llm_call(
            provider=self.provider,
            model=self.model,
            seconds=time.monotonic() - started,
            usage=data.get("usage"),
        )

    # --- SDK exceptions → the harness provider error hierarchy ----------------

    def _mapped_errors(self):
        """A context manager mapping ``openai`` SDK exceptions onto the harness's typed errors.

        The engine and tools catch `ProviderConnectionError` / `ProviderAPIError` and its
        subclasses, and the image-error relay digs the real cause out of a `ProviderAPIError`'s
        ``body`` — so the SDK's own exception zoo is normalized here, once, into that contract.
        """
        return sdk_error_context(self._openai)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> OpenAIProvider:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def sdk_error_context(openai):
    """A context manager mapping ``openai`` SDK exceptions onto the harness `ProviderError`s.

    The shared seam the model adapter *and* the image/audio platform tools use, so an SDK
    status error always arrives as a `ProviderAPIError` carrying the response body — which the
    media error relay (`provider_error_message`) digs the real cause out of.
    """
    return _ErrorMapper(openai)


class _ErrorMapper:
    """Translate ``openai`` SDK exceptions into the harness `ProviderError` hierarchy."""

    def __init__(self, openai) -> None:
        self._openai = openai

    def __enter__(self) -> _ErrorMapper:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is None:
            return False
        openai = self._openai
        if isinstance(exc, openai.APIConnectionError):
            # Covers APITimeoutError too — DNS/TCP/TLS/timeout, nothing reached the model.
            raise ProviderConnectionError(f"Could not reach the provider: {exc}") from exc
        if isinstance(exc, openai.APIStatusError):
            raise _from_status_error(exc) from exc
        if isinstance(exc, openai.APIError):
            # A non-status SDK error: the response arrived but could not be parsed — a truncated
            # body, malformed JSON, or a schema mismatch (`openai.APIResponseValidationError` lands
            # here — it is an APIError, not an APIStatusError). This is the transient
            # unparseable-response class (issue #259), so it maps to the retryable
            # `ProviderResponseError`; the engine re-requests it before giving up.
            raise ProviderResponseError(str(exc)) from exc
        if isinstance(exc, json.JSONDecodeError):
            # httpx's ``response.json()`` raises this on a truncated / non-JSON 200 body, and the
            # SDK lets it propagate raw — it is exactly the "EOF while parsing a value" fault this
            # issue names, so it too maps to the retryable `ProviderResponseError` (issue #259).
            raise ProviderResponseError(
                f"Provider returned an unparseable response body: {exc}"
            ) from exc
        return False  # not an SDK error — let it propagate unchanged


def _from_status_error(exc) -> ProviderError:
    """An ``openai.APIStatusError`` mapped to the right typed `ProviderAPIError` subclass."""
    status = exc.status_code
    body = _body_text(exc)
    message = getattr(exc, "message", None) or str(exc)
    if status in (401, 403):
        return ProviderAuthError(
            f"Provider rejected the API key (HTTP {status}).", status_code=status, body=body
        )
    if status == 429:
        return ProviderRateLimitError(
            "Provider rate-limited the request (HTTP 429).",
            status_code=status,
            body=body,
            retry_after=_retry_after(exc),
        )
    # Carry the provider's own message so the image/audio tools can relay the true cause.
    return ProviderAPIError(message, status_code=status, body=body)


def _body_text(exc) -> str:
    """The raw error body text from an SDK status error, for `ProviderAPIError.body`.

    The image/audio error relay (`provider_error_message`) parses this for the real cause, so
    it must be the response *text* (the JSON envelope), preferring the live response over the
    SDK's pre-parsed ``body``.
    """
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            return response.text
        except Exception:  # noqa: BLE001 - a body we can't read degrades to empty, never crashes
            pass
    body = getattr(exc, "body", None)
    return str(body) if body is not None else ""


def _retry_after(exc) -> float | None:
    """The ``Retry-After`` seconds hinted on a 429 response, if present and numeric."""
    response = getattr(exc, "response", None)
    if response is None:
        return None
    raw = response.headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None
