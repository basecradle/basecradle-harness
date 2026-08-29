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

Cache affinity — ``x-grok-conv-id`` as gRPC metadata (issues #431, #433)
------------------------------------------------------------------------
xAI's prompt cache is **per-server**, so a repeated prefix only pays out when the next call lands on
the server that holds it. xAI's remedy is a stable conversation id, and their prompt-caching guide
spells it out per surface: the ``x-grok-conv-id`` HTTP header on Chat Completions,
``prompt_cache_key`` in the Responses body, and — for the gRPC API this adapter speaks — *"pass
``x-grok-conv-id`` as gRPC metadata to enable sticky routing for cache reuse."* The cost of sending
nothing was measured, not theorized: @briggs hit 0.2–18% on a byte-stable ~210 K-token prefix where
every other adapter was earning 92–99%.

**The near miss is worth writing down** (issue #433, found by the NOC as basecradle#512). Version
0.110.0 bound the key to ``chat.create(conversation_id=...)`` — which the SDK *accepts* and then
**never puts on the wire**: in ``xai-sdk`` 1.19.0 ``chat.create`` declares ``conversation_id`` as
its own keyword and hands it to the `Chat` **beside** the request settings rather than into them,
and its sole consumer there is the OpenTelemetry span attribute ``gen_ai.conversation.id``; no
request proto in the wheel carries such a field at all.
Every test passed, no log line changed, and the discount stayed unearned. **A key a Python
signature accepts is not a key on the wire** — which is why the tests for this stand up a real gRPC
server and read back the metadata the SDK actually sent (`tests/test_xai_sdk_wire.py`).

So the key now rides where xAI says to put it: gRPC **call metadata**, which is HTTP/2 headers by
another name. The one constraint the design has to absorb is that ``xai_sdk`` fixes that metadata
**per Client** — it is baked into the channel's ``_APIAuthPlugin`` (TLS) or ``AuthInterceptor``
(insecure) at construction, and the SDK's stub calls pass no per-call ``metadata=``. There is
therefore no per-call seam to reach, so the adapter **rebuilds its client when the bound
conversation changes** (`_bound_client`): cheap, because a gRPC channel connects lazily, and rare,
because every turn of a session binds the same key. `bind_conversation` takes the harness **session
id**; unbound, nothing is sent — never a fabricated value, which would read as a fresh conversation
to xAI on every call. The reasoning, and why OpenRouter's rejected session pin (issue #372) is the
opposite situation rather than a precedent against this, lives in `_caching`.

Stateless per turn: the full conversation is sent every call and the harness owns history.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Sequence
from typing import Any

from basecradle_harness._caching import AUTOMATIC
from basecradle_harness._context import is_context_overflow
from basecradle_harness._exceptions import (
    ProviderAuthError,
    ProviderBillingError,
    ProviderConnectionError,
    ProviderContextLengthError,
    ProviderError,
    ProviderPayloadTooLargeError,
    ProviderRateLimitError,
    ProviderResponseError,
)
from basecradle_harness._faults import is_out_of_funds, is_too_large
from basecradle_harness._messages import ImageContent, Message, ToolCall, ToolSpec
from basecradle_harness._observability import log_llm_call, serving_endpoint, token_counts
from basecradle_harness._openai_wire import format_citations

_log = logging.getLogger("basecradle_harness")

#: This adapter's single native (gRPC) surface — declared for the SDK-scoped surface contract
#: (issue #163); ``AI_SDK_SURFACE`` is left unset for a single-surface SDK.
SURFACES = ("native",)
DEFAULT_SURFACE = "native"
#: The endpoint vendor this adapter reaches — the native SDK talks only to xAI, so unlike the
#: `openai` adapter (which serves three) it is a class constant, not a constructor arg. It rides
#: the per-call log line as ``provider=xai``.
PROVIDER = "xai"

#: xAI's cache-affinity routing key, spelled the way their docs spell it for **this** surface
#: (issue #433): *"pass ``x-grok-conv-id`` as gRPC metadata to enable sticky routing for cache
#: reuse."* gRPC metadata *is* HTTP/2 headers, so this is the same header the REST surfaces take —
#: which is exactly why it belongs in the call's metadata and not in the request body, where the
#: 1.19.0 wheel has no field to put it in (see the module docstring).
CONVERSATION_METADATA_KEY = "x-grok-conv-id"


def _close_client(client: Any) -> None:
    """Release a client's gRPC channel, if it is the kind of client that has one to release.

    Duck-typed because an injected client (the test seam, a library caller's own) need not offer
    ``close`` at all — and **best-effort**, because closing is cleanup: a channel that refuses to
    close (a pending call, a library caller's own throwing ``close``) is a leaked socket, and
    letting that escape would turn a leaked socket into a dead wake.
    """
    close = getattr(client, "close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception as exc:  # noqa: BLE001 - cleanup; a failure to release must not fail the caller
        _log.warning("Could not close an xAI client's gRPC channel: %s", exc)


def _fits_grpc_metadata(value: str) -> bool:
    """Can `value` ride as an ASCII gRPC metadata value (issue #433)?

    grpc encodes a non-``-bin`` metadata value as an HTTP/2 header value and rejects anything
    outside printable ASCII. The check lives here, on the way *in*, so a session id that could
    never reach the wire is dropped at bind time instead of raising inside the model call.
    """
    return all(" " <= ch <= "~" for ch in value)


def require_xai_sdk():
    """Import and return the ``xai_sdk`` package, or raise a clear "no LLM, by design" error.

    The core depends on **no** vendor SDK — an ``AI_SDK=xai-sdk`` agent installs the extra
    (``pip install 'basecradle-harness[xai-sdk]'``). With it absent the harness genuinely cannot
    reach a model, so this fails loud and actionable at provider construction rather than letting a
    bare ``ModuleNotFoundError`` surface from inside a wake.
    """
    try:
        import xai_sdk  # lazy: the core must import without the vendor SDK
        import xai_sdk.tools  # submodule (Agent Tools) is not auto-imported by __init__
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
            ``temperature=0.2``). ``model``, ``messages``, ``tools`` always take precedence — and so
            does ``conversation_id`` once something has bound one (`bind_conversation`). In a
            deployment an operator never reaches this: ``conversation_id`` is harness-owned, so
            `_basecradle._split_model_params` strips it out of ``model_params.json`` with a WARNING
            first — a single static value would pin every session on the box to one xAI server,
            defeating the affinity it looks like it is asking for. The precedence here is what keeps
            a library caller who passes one directly from silently overriding a bound session.
    """

    #: How xAI reaches its prompt cache (issue #277): **automatically**, with nothing on the wire —
    #: the proto reports the hit back as ``cached_prompt_text_tokens``, which is already on the
    #: per-call log line. The engine places no breakpoint. This adapter is direct-to-vendor, so
    #: unlike the router-fronting adapters it carries no routed-*model* caveat: xAI is the only
    #: endpoint it can reach, and grok's cache is automatic. It does carry a routed-*server* one —
    #: the cache is per-server, which `bind_conversation` addresses (issue #431).
    cache_mode = AUTOMATIC

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
        self.provider = PROVIDER
        #: The input-token count xAI reported for this adapter's most recent call — the exact,
        #: free, tokenizer-free trigger the context budget compacts on (issue #276). ``None``
        #: until the first call answers.
        self.last_tokens_in: int | None = None
        self._builtin_tools = list(builtin_tools)
        #: The conversation this adapter's next calls belong to — xAI's per-server cache-affinity
        #: routing key (issues #431, #433), set by `bind_conversation` and ``None`` until something
        #: binds one. A library caller driving the engine with no `Session` never binds, and then an
        #: operator's own ``conversation_id`` in ``model_params.json`` (if any) is what rides.
        self._conversation: str | None = None
        #: What `self._client` was built with, kept so it can be **rebuilt** when the bound
        #: conversation changes — the SDK fixes a channel's gRPC metadata at construction, so that
        #: is the only seam the key can reach (issue #433). ``None`` for an injected client: there
        #: is nothing to rebuild it from, and rebuilding someone else's client is not ours to do.
        self._client_kwargs: dict[str, Any] | None = None
        #: The bound conversation `self._client` has already been **reconciled against** — normally
        #: the key its metadata carries, and on a rebuild that failed, the key it gave up on. Either
        #: way it is what makes the work happen on a genuine change and not once per turn.
        self._client_conversation: str | None = None
        #: The last conversation refused as unable to ride the wire, so the warning is emitted once
        #: per bad key rather than once per turn (a session rebinds before every turn).
        self._refused_conversation: str | None = None
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
            self._client_kwargs = kwargs
            self._client = self._xai.Client(**kwargs)

    def chat(self, messages: Sequence[Message], tools: Sequence[ToolSpec] | None = None) -> Message:
        """Run one model turn through the native SDK and return the assistant's reply."""
        chat_mod = self._xai.chat
        payload: dict[str, Any] = dict(self._default_params)
        payload["model"] = self.model
        payload["messages"] = [self._to_wire(m, chat_mod) for m in messages]
        if self._conversation:
            # Telemetry only, and named here so nobody re-derives 0.110.0's mistake from the code:
            # `xai-sdk` 1.19.0 takes `conversation_id` as its own keyword, keeps it beside the
            # request settings, and spends it on one OpenTelemetry span attribute
            # (`gen_ai.conversation.id`) — it reaches no request proto
            # and therefore no server (issue #433 / basecradle#512). It is still the *right* value
            # for that attribute, so it stays; what actually earns the cache is the
            # `x-grok-conv-id` metadata on `_bound_client`'s channel. Harness-owned like
            # `model`/`messages`/`tools`, and for a sharper reason — a *static* `conversation_id` in
            # `model_params.json` would pin every session on the box to one server, the anti-pattern
            # this exists to prevent. Unbound, whatever the operator set is left exactly as it is.
            payload["conversation_id"] = self._conversation
        # Function tools and the opted-in server-side built-ins (search, code execution) share
        # one ``tools`` list: all are native ``chat_pb2.Tool`` protos (issue #171 — Agent Tools).
        wire_tools = [chat_mod.tool(t.name, t.description, t.parameters) for t in tools or ()]
        wire_tools.extend(self._agent_tools())
        if wire_tools:
            payload["tools"] = wire_tools
        started = time.monotonic()
        # The client whose channel carries this session's `x-grok-conv-id` — the affinity key's
        # only route to the wire on this SDK (issue #433).
        client = self._bound_client()
        with self._mapped_errors():
            conversation = client.chat.create(**payload)
            response = conversation.sample()
        # The native response carries usage as a proto (attributes, not keys); `log_llm_call`
        # reads either shape, so the gRPC path logs the same line as the HTTP ones — token counts
        # and the cached-prompt count (xAI spells it `cached_prompt_text_tokens`). A fake client
        # (the seam tests) whose response has no `usage` simply logs no usage fields.
        usage = getattr(response, "usage", None)
        # Remember what we just logged: the context budget triggers on the *provider's* count, so
        # the same read that writes the log line feeds the compaction decision (issue #276).
        self.last_tokens_in = token_counts(usage).get("tokens_in")
        log_llm_call(
            provider=self.provider,
            model=self.model,
            seconds=time.monotonic() - started,
            usage=usage,
            # Asked of every adapter, answered by the ones that can: this SDK reaches xAI directly,
            # so the vendor *is* the endpoint — there is no upstream to name, and the field is
            # omitted rather than faked with a restatement of `provider=xai`.
            endpoint=serving_endpoint(response),
            # xAI states the charge natively, but in *ticks* (1 tick = 1e-10 USD) — so the dollars
            # come from the SDK's own accessor, never harness arithmetic and never a price table:
            # the constant is xAI's to change. It reports ``None`` when the server named no cost, so
            # an unreported call logs nothing rather than a fabricated ``cost=0``; an SDK too old to
            # carry the property does the same.
            cost=getattr(response, "cost_usd", None),
        )
        return self._from_wire(response)

    def bind_conversation(self, conversation: str | None) -> None:
        """Route this adapter's next calls to the server holding `conversation`'s prefix (#431).

        The cache-affinity capability (`_caching.bind_conversation`). The harness binds the session
        id before each turn; ``None`` (or an empty string) clears it, and the next call then sends
        **no** ``x-grok-conv-id`` at all rather than inventing one — a fabricated id is a new
        conversation to xAI on every call, which buys a guaranteed miss instead of a lucky one.

        A key gRPC could not carry is **refused here rather than raised there** (issue #433). ASCII
        metadata values are printable ASCII on the wire, and grpc rejects anything else — at *call*
        time, inside `chat`, where it would fail the model call and the whole wake. Affinity is an
        optimization and must never cost a wake, so an unusable key is logged and dropped, leaving
        this adapter exactly as well off as it was before #431: unbound, and lucky.
        """
        key = conversation or None
        if key is not None and not _fits_grpc_metadata(key):
            if key != self._refused_conversation:
                # Once per bad key, not once per turn: a session rebinds the same id before every
                # turn, so an unconditional warning here would repeat for the life of the process.
                _log.warning(
                    "Not binding %r for xAI cache affinity: a gRPC metadata value must be "
                    "printable ASCII. Calls will run unbound (no x-grok-conv-id), which costs the "
                    "per-server cache hit but never a wake.",
                    conversation,
                )
                self._refused_conversation = key
            key = None
        else:
            self._refused_conversation = None
        self._conversation = key

    def _bound_client(self) -> Any:
        """The client whose gRPC metadata carries the currently-bound conversation (issue #433).

        ``xai_sdk`` fixes a client's metadata **at construction** — it is closed over by the
        channel's ``_APIAuthPlugin`` (TLS) or ``AuthInterceptor`` (insecure), and the SDK's stub
        calls pass no per-call ``metadata=`` — so there is no per-call seam, and switching the key
        means switching the client. That is cheaper than it reads: a gRPC channel connects **lazily**,
        so building one costs an object, not a handshake, and the handshake it does eventually cost
        is amortized over every turn of the session that asked for it.

        Rebuilt only on a genuine **change**, which is what keeps that true: every turn of a session
        binds the same id, so the steady state is one client per session and nothing per turn. An
        **injected** client (the test seam, a library caller supplying their own) is returned
        untouched — there are no kwargs to rebuild it from, and silently replacing a client someone
        handed us would discard whatever they configured it with.
        """
        if self._client_kwargs is None or self._client_conversation == self._conversation:
            return self._client
        kwargs = dict(self._client_kwargs)
        if self._conversation:
            kwargs["metadata"] = ((CONVERSATION_METADATA_KEY, self._conversation),)
        stale = self._client
        try:
            fresh = self._xai.Client(**kwargs)
        except Exception as exc:  # noqa: BLE001 - affinity is an optimization; never break a wake
            # This is the one part of the affinity path that runs inside `chat`, so it sits outside
            # both guards that used to cover it: `_caching.bind_conversation`'s blanket try/except
            # (which only wraps the bind) and `_mapped_errors` (which only classifies gRPC faults).
            # A raw failure here would kill the wake for an optimization. Worse, it would kill
            # *every* wake: the session rebinds the same id before every turn, so an unguarded
            # rebuild would re-attempt the identical failure forever. So the adapter **gives up on
            # this conversation** — marking it reconciled below is what makes that once, not
            # per-turn — and keeps running on the client it already has, unbound and lucky, exactly
            # where #431 found it. A *different* conversation gets its own fresh attempt.
            _log.warning(
                "Could not rebuild the xAI client for cache affinity (%s); continuing unbound on "
                "the existing client. Calls will run without x-grok-conv-id, which costs the "
                "per-server cache hit but never a wake.",
                exc,
            )
            self._client_conversation = self._conversation
            return stale
        # Publish the new client *before* releasing the old one. The other order leaves a window
        # where a throwing `close` strands `fresh` unreferenced with its channel still open — the
        # very per-switch socket leak this close exists to prevent, reached through the other door.
        self._client = fresh
        self._client_conversation = self._conversation
        _close_client(stale)
        return fresh

    def context_limit(self) -> int | None:
        """This model's context ceiling, straight from xAI — the `ContextBudget` capability (#276).

        The cleanest answer of the three adapters: xAI's own model metadata carries the number
        (``LanguageModel.max_prompt_length`` on the gRPC proto), so there is nothing to infer and no
        table to rot. One cheap gRPC call, made lazily and at most once per process.

        ``None`` on any failure or a model that reports no length — the budget then falls to its
        conservative floor. A metadata read must never break a wake.
        """
        try:
            model = self._client.models.get_language_model(self.model)
        except Exception as exc:  # noqa: BLE001 - degrade to the floor; never break a wake
            _log.warning("Could not read %s's context limit from xAI: %s", self.model, exc)
            return None
        length = getattr(model, "max_prompt_length", None)
        if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
            return None
        return length

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
        _close_client(self._client)

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
            import grpc  # lazy: only needed to classify, only on the error path
        except ModuleNotFoundError:  # pragma: no cover - grpc ships with xai-sdk
            return False
        if not isinstance(exc, grpc.RpcError):
            return False
        code = exc.code() if callable(getattr(exc, "code", None)) else None
        detail = exc.details() if callable(getattr(exc, "details", None)) else str(exc)
        message = f"xAI gRPC error ({getattr(code, 'name', code)}): {detail}"
        detail = detail or ""
        if is_context_overflow(detail):
            # The wall (issue #276): the prompt was over grok's context window — gRPC's
            # INVALID_ARGUMENT, the HTTP 400's analogue. Deterministic, so the session compacts and
            # retries the turn once rather than re-sending a request that fails identically forever.
            raise ProviderContextLengthError(message, status_code=400, body=detail) from exc
        if code == grpc.StatusCode.UNAUTHENTICATED:
            raise ProviderAuthError(message, status_code=401) from exc
        if code == grpc.StatusCode.RESOURCE_EXHAUSTED:
            # gRPC overloads RESOURCE_EXHAUSTED across three faults with three different remedies, so
            # the *detail string* decides — never the bare code (issue #336). This is the exact
            # misclassification the 2026-07-21 @briggs incident exposed: a client-side "message
            # larger than max" was read as a rate limit and re-driven 51 times.
            if is_too_large(detail):
                # The ``xai-sdk``'s own 20 MiB channel cap, computed *client-side* before the wire:
                # the request body is too large. Deterministic — the identical bytes are rejected
                # identically — so it is reported once, never retried, and the file is never modified
                # (decision 1). `status_code=413`: the HTTP analogue, so a caller reading the status
                # sees "payload too large" regardless of the transport.
                raise ProviderPayloadTooLargeError(message, status_code=413, body=detail) from exc
            if is_out_of_funds(detail):
                # Out of xAI credit — the account-blocked class. **Defensive**: xAI's exact
                # out-of-credit gRPC shape could not be confirmed from its published docs (issue
                # #336 says to match defensively and say so), so this rests on the detail phrasing;
                # an unrecognized out-of-credit wording falls through to the rate-limit default
                # below, which is the safe direction (a retry, not a false outage report).
                # `status_code=402`: the HTTP Payment-Required analogue.
                raise ProviderBillingError(message, status_code=402, body=detail) from exc
            # A genuine rate limit — heals with time, so transient, exactly as before.
            raise ProviderRateLimitError(message, status_code=429) from exc
        if code in (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED):
            raise ProviderConnectionError(message) from exc
        if code in (grpc.StatusCode.INTERNAL, grpc.StatusCode.DATA_LOSS):
            # A broken/undecodable response payload — gRPC's analogue of the truncated-JSON class
            # (issue #259): the call reached the endpoint and came back corrupt. Map it to the
            # retryable `ProviderResponseError` so the engine re-requests it, the same capability
            # class as the OpenAI/OpenRouter parse failures — a dropped wake, not a config bug.
            raise ProviderResponseError(message) from exc
        # Anything else — an INVALID_ARGUMENT that is not a context overflow (gRPC's deterministic-400
        # analogue), a PERMISSION_DENIED, etc. — stays a plain `ProviderError` and **propagates**. A
        # generic malformed request is almost always a fixable harness/config defect, not a permanent
        # property of the peer's content, so reporting-and-marking-handled would lose the peer's
        # message once the config is fixed; propagating leaves it re-drivable (issue #336; CLAUDE.md →
        # Provider Capabilities, "a bad model_params.json key propagates on the first raise").
        raise ProviderError(message) from exc
