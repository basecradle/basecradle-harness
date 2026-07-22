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
import logging
import os
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from basecradle_harness._caching import AUTOMATIC
from basecradle_harness._context import is_context_overflow
from basecradle_harness._exceptions import (
    ProviderAPIError,
    ProviderAuthError,
    ProviderBillingError,
    ProviderConnectionError,
    ProviderContextLengthError,
    ProviderError,
    ProviderPayloadTooLargeError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderServerError,
)
from basecradle_harness._faults import is_out_of_funds
from basecradle_harness._messages import Message, ToolSpec
from basecradle_harness._observability import (
    log_llm_call,
    reported_cost,
    serving_endpoint,
    token_counts,
)
from basecradle_harness._openai_wire import (
    builtin_to_responses,
    chat_message_to_wire,
    chat_tool_to_wire,
    function_tool_to_responses,
    message_from_chat,
    message_from_responses,
    message_to_input,
)

_log = logging.getLogger("basecradle_harness")

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
        extra_headers: Headers sent on **every** request, as the SDK client's ``default_headers``.
            The header-side twin of ``extra_body``, and for the same reason: an endpoint may put a
            fact behind a request header rather than a body field. Today's use is OpenRouter's
            ``X-OpenRouter-Metadata``, which is what makes it state the endpoint it actually routed
            to (issue #280). The config layer decides what to send; the adapter just carries it.
        code_container: An optional callback supplying the ``container`` config for the
            ``code_interpreter`` built-in, evaluated **per turn** (the container handle changes
            as the Asset bridge stages files / pins a session — see `_code.py`). Returns a
            container id string, a container dict, or ``None``. When absent (or it returns
            ``None``) the built-in falls back to ``{"type": "auto"}``. The adapter stays
            BaseCradle-agnostic — it just asks "what container?"; the bridge answers.
        default_params: Extra body parameters sent on every call (e.g. ``temperature=0.2``).
            ``model``, the input/messages, and ``tools`` always take precedence.
    """

    #: How this adapter's endpoints reach their prompt cache (issue #277). Every endpoint this one
    #: adapter is aimed at — OpenAI, xAI, and OpenRouter's GLM endpoints — caches a repeated prefix
    #: **automatically**, with nothing on the wire, so the engine places no breakpoint and the
    #: caching that already works (verified live: a `cached_tokens: 238277` hit) is untouched.
    #:
    #: The stated exception, so it is not a silent trap: pointed at an **explicit-cache model
    #: through a router** (``anthropic/claude-*`` via OpenRouter), this declaration is wrong — that
    #: agent would cache nothing and pay full freight. It is not a live cell (no fleet agent runs
    #: one) and closing it means resolving the mode from the *routed* model rather than the adapter,
    #: which is the natural shape of the native Anthropic adapter this capability exists to unblock.
    cache_mode = AUTOMATIC

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
        extra_headers: Mapping[str, str] | None = None,
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
        #: The input-token count the endpoint reported for this adapter's most recent call — the
        #: exact, free, tokenizer-free trigger the context budget compacts on (issue #276). ``None``
        #: until the first call answers.
        self.last_tokens_in: int | None = None
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
            # Sent on every request the client makes. The endpoint this one adapter is aimed at
            # decides whether there is anything to send — the config layer answers that, so the
            # adapter stays vendor-neutral (the header seam, exactly as `extra_body` is the body one).
            default_headers=dict(extra_headers) if extra_headers else None,
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
        """The one INFO line this call earns: provider, endpoint, model, duration, tokens, cost.

        Both surfaces report usage, under different names (Responses ``input_tokens`` vs Chat
        ``prompt_tokens``) — `log_llm_call` reads either, so this one call site serves both. Only
        a call that *returned* is logged; a call that raised is the error path's story to tell
        (the engine logs the retry/give-up), and timing a failure as if it were a completion
        would be a lie.

        The serving **endpoint** and the **cost** are capability reads, not vendor branches: this
        one adapter is aimed at three endpoints, and the *response* is what says whether either
        fact exists. Pointed at OpenRouter it comes back naming the upstream that served the call
        and what it charged (the ``openai`` SDK's models keep unmodeled fields, so both survive
        `model_dump`); pointed at OpenAI or xAI it says neither, and the fields are simply absent.
        """
        usage = data.get("usage")
        # Remember what we just logged: the context budget triggers on the *provider's* count, so
        # the same usage read that writes the log line feeds the compaction decision (issue #276).
        # Both surfaces are covered for free — `token_counts` already knows every spelling.
        self.last_tokens_in = token_counts(usage).get("tokens_in")
        log_llm_call(
            provider=self.provider,
            model=self.model,
            seconds=time.monotonic() - started,
            usage=usage,
            endpoint=serving_endpoint(data),
            cost=reported_cost(usage),
        )

    def context_limit(self) -> int | None:
        """This model's context ceiling, if the endpoint this adapter is aimed at states one (#276).

        A **capability read, not a vendor branch** — this one adapter serves three endpoints and the
        *endpoint* is what decides whether the fact exists:

        - Pointed at **OpenRouter**, the models API states a context length, and the `openai` SDK's
          models keep unmodeled fields through `model_dump()`, so it survives and is read here.
        - Pointed at **OpenAI**, the models API states id/created/owned_by and *nothing about
          context*. So this honestly returns ``None`` and the budget falls to its conservative floor.
          That is the deliberate cost of refusing a static model→limit table (issue #276,
          requirement 2): a table would answer today and lie after the next model launch, silently.
          An OpenAI agent that wants its real 400 K window sets `HARNESS_MAX_CONTEXT_TOKENS`.

        Never fatal: any failure degrades to ``None``, and the wake runs exactly as before.
        """
        try:
            model = self._client.models.retrieve(self.model)
        except Exception as exc:  # noqa: BLE001 - degrade to the floor; never break a wake
            _log.warning("Could not read %s's context limit from the provider: %s", self.model, exc)
            return None
        data = model.model_dump() if hasattr(model, "model_dump") else {}
        if not isinstance(data, Mapping):
            return None
        # The spellings an OpenAI-compatible endpoint uses for the same fact. OpenAI itself uses
        # none of them, which is the honest answer, not a gap to paper over.
        for key in ("context_length", "context_window", "max_context_length"):
            value = data.get(key)
            if not isinstance(value, bool) and isinstance(value, int) and value > 0:
                return value
        return None

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
    """An ``openai.APIStatusError`` mapped to the right typed `ProviderAPIError` subclass.

    The failure taxonomy (issue #336) is mapped here by the *nature* of the fault, never by the
    vendor — this one adapter serves OpenAI, xAI, and OpenRouter, so what it reads is the wire signal,
    not who sent it. Beyond the pre-existing classes (context-overflow, auth, rate-limit, 5xx), two
    **reported** classes are distinguished:

    - **Out of funds** — OpenAI signals it as a **429 whose ``error.type``/``code`` is
      ``insufficient_quota``**, the one 429 that is *not* a rate limit. It heals only when a human
      funds the account, so it is `ProviderBillingError`, not a rate limit; the wake reports it and
      debounces rather than retrying. The **structured code is checked first and status-independently**
      (so an endpoint that signals out-of-funds as a 401/403 — e.g. xAI-via-``openai``, whose shape is
      unconfirmed — is not swallowed as an auth error), with a text fallback scoped to the 429 body.
    - **Payload too large** — a **413** that is not a context overflow: the request body exceeded the
      endpoint's accept limit. Deterministic and content-shaped, so `ProviderPayloadTooLargeError`
      (reported, the original never modified — issue #336, decision 1), not a retry.

    A *generic* malformed-request 400/422 is deliberately **not** in the taxonomy: it is almost always
    a fixable harness/config defect (a bad ``model_params.json`` key, a serialization bug), not a
    permanent property of the peer's content — so it stays a plain `ProviderAPIError` and **propagates**
    (CLAUDE.md → Provider Capabilities: "a bad model_params.json key propagates on the first raise").
    Reporting it and marking the item handled would lose the peer's message the moment the config is
    fixed; propagating leaves the message re-drivable. See issue #336's completion notes.
    """
    status = exc.status_code
    body = _body_text(exc)
    message = getattr(exc, "message", None) or str(exc)
    if status in (400, 413) and is_context_overflow(f"{message} {body}"):
        # The wall (issue #276): the transcript outgrew the model's context window. Deterministic —
        # every later wake would rebuild the same over-long request and fail identically — so it is
        # classed apart from every other 400 and the session compacts and retries the turn once.
        return ProviderContextLengthError(message, status_code=status, body=body)
    if _error_code(body) == "insufficient_quota":
        # Out of funds — the account-blocked class. The **structured** code is authoritative and
        # status-independent, so a 401/403/429 all resolve to billing here rather than being
        # swallowed as auth/rate-limit below (issue #336). No false-positive risk: this is the
        # vendor's own machine-readable code, not a message-text guess.
        return ProviderBillingError(message, status_code=status, body=body)
    if status in (401, 403):
        return ProviderAuthError(
            f"Provider rejected the API key (HTTP {status}).", status_code=status, body=body
        )
    if status == 429:
        # A 429 without the structured quota code: a genuine rate limit (transient), unless the body
        # text names an out-of-funds cause — the defensive fallback for an endpoint whose body shape
        # is not confirmed (xAI-via-``openai``). The text match is scoped to the 429 body so it can
        # never re-read a 4xx that isn't rate-limit-shaped as an outage.
        if is_out_of_funds(f"{message} {body}"):
            return ProviderBillingError(message, status_code=status, body=body)
        return ProviderRateLimitError(
            "Provider rate-limited the request (HTTP 429).",
            status_code=status,
            body=body,
            retry_after=_retry_after(exc),
        )
    if status == 413:
        # A 413 that is not a context overflow (checked above): the request body was simply too
        # large. Deterministic and file-shaped — reported, never retried, never modified (issue #336).
        return ProviderPayloadTooLargeError(message, status_code=status, body=body)
    if status >= 500:
        # The provider fell over on its own side — transient, so the engine re-requests it
        # (issue #284). The SDK also retries 5xx internally; this class is what makes the policy
        # *uniform* across adapters instead of a property of whichever SDK an agent happens to run.
        return ProviderServerError(
            f"Provider failed on its own side (HTTP {status}).", status_code=status, body=body
        )
    # Carry the provider's own message so the image/audio tools can relay the true cause — and so a
    # generic malformed-request 400/422 propagates (see the docstring) rather than being reported.
    return ProviderAPIError(message, status_code=status, body=body)


def _error_code(body: str) -> str | None:
    """The ``error.type`` (or ``error.code``) from an OpenAI-shaped error body, if present.

    OpenAI's out-of-funds 429 carries ``{"error": {"type": "insufficient_quota", ...}}`` — a
    structured signal, read here so the billing class does not rest on a message-text match. Any
    non-JSON or unshaped body yields ``None`` and the caller falls back to the text heuristic.
    """
    if not body:
        return None
    try:
        data = json.loads(body)
    except ValueError:
        return None
    error = data.get("error") if isinstance(data, dict) else None
    if not isinstance(error, dict):
        return None
    code = error.get("type") or error.get("code")
    return code if isinstance(code, str) else None


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
