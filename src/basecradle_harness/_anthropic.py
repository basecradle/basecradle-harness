"""The native Anthropic adapter — Claude, direct, over Anthropic's own ``anthropic`` SDK (issue #725).

The fifth `Provider` adapter. It reaches Claude through Anthropic's first-party Python SDK
(``anthropic`` on PyPI) and its Messages API — no OpenAI-compatibility shim, no harness-owned HTTP.
Selected by ``AI_PROVIDER=anthropic`` + ``AI_SDK=anthropic``; the key is ``AI_API_KEY``.

Why direct
----------
The fleet calls a first-party vendor it holds an account with **direct**, through that vendor's own
adapter, and keeps OpenRouter for open-weight models with many hosts (settled 2026-10-07). Claude
reached through a router is also the one cell where the router adapters' ``automatic`` cache mode is
wrong (issue #702); this adapter is the direct route, so that gap does not apply to it.

The credential, and what the SDK is never allowed to read for itself
--------------------------------------------------------------------
The SDK resolves a missing credential from the environment — ``ANTHROPIC_API_KEY``, then
``ANTHROPIC_AUTH_TOKEN``, a profile on disk, workload-identity variables — and a missing host from
``ANTHROPIC_BASE_URL``. Either fallback would silently borrow somebody else's account or send this
agent's key somewhere it was not configured to go, which is exactly what this fleet's auth fences
exist to stop. So the adapter always hands the SDK **both**: the key it was given (``AI_API_KEY``)
and a host (``AI_BASE_URL``, else `DEFAULT_BASE_URL`); a build with no key is a configuration fault
at startup, never an SDK that went looking. The one variable the SDK reads unconditionally,
``ANTHROPIC_CUSTOM_HEADERS``, adds headers to every request and cannot be refused through its API,
so a box that sets it gets a WARNING naming it. The SDK client is held `Sealed` (issue #599): it
keeps the key in its own ``__dict__``.

Explicit caching — the fleet's first ``cache_mode = "explicit"``
---------------------------------------------------------------
Anthropic caches only a prefix the client **marks**, so this is the adapter #277 built the engine's
breakpoint for. The engine stamps `Message.cache_anchor` on the last turn of the frozen transcript
(`_caching.anchor_cacheable_prefix`); here that becomes ``cache_control: {"type": "ephemeral"}`` on
the anchored turn's last block — a system block when the anchor is the charter. Three placements are
the adapter's own, because only the adapter knows the wire's positions (issues #726, #729). The four
breakpoints Anthropic allows are spent as **anchor + tail + two chained**:

- **The tail, on every step.** The anchor caches what *later wakes* read; everything after it — the
  per-wake brief, the new turn, the wake's own growing tool chain — is volatile across wakes and
  byte-stable across the steps of one wake, which re-sends it on every call. So the request's last
  block that can carry one gets a breakpoint too: step *k* writes its tail and step *k + 1* reads
  it, paying the write rate only on its own growth (Anthropic's documented pattern for an agentic
  loop). No later wake can read a tail entry, since the next wake's prefix lacks this wake's brief;
  the next step of this wake can, and that read is the point. It goes on step 1 too: the step 1 →
  step 2 read is the largest (the whole brief), and a wake cannot know at step 1 whether a step 2 is
  coming. A single-step wake pays a write premium on its brief and turn for that (0.25× at the
  five-minute TTL, 1× at the hour), and the decision to pay it is recorded on issue #729. The read
  holds while one step adds fewer than `LOOKBACK` positions: a step past that (a reply of many
  server-tool calls or citation blocks, which Anthropic does not count as runs) leaves the next
  step's tail blind, and that step falls back to the anchor's entry — a rewrite, never a failure.
- **No anchor, the tool list.** On a first wake the frozen transcript is empty (the charter rides
  the per-wake brief), so the engine anchors nothing. The stable/volatile boundary is then the end
  of ``tools``, which renders first, and the last tool definition carries the breakpoint — the one
  entry a first wake writes that the *next* wake can read.
- **The lookback chain.** A breakpoint finds an earlier entry only within `LOOKBACK` (20) positions
  of itself, and the entry a wake can find is the one the previous wake wrote at *its* anchor. A
  wake that added 20 positions or more (about seven sequential tool steps) would leave the next
  wake's anchor blind, rewriting the whole prefix at the write rate. So `CHAINED` (two) more
  breakpoints are chained back from the anchor, `LOOKBACK` apart, reaching 60 positions; a hit is a
  read and the write is charged once, so they cost nothing (`AnthropicProvider._place_breakpoints`).
  A step whose model call ran a server tool replays a fourth position, its record's note
  (`_server_tools`, issue #727), so on a wake heavy with server tools the reach is about 15 steps.

Every placement is on the request only: copy-on-write, never persisted, and never counted by the
thinking-replay digest (`_Stream` reads every block without its ``cache_control``). Three properties
make the marks pay out, and each one fails silently when broken:

- **The render order is tools → system → messages**, so the tool list is part of every cached
  prefix and must be byte-stable for a given agent: server tools first in a fixed order, then the
  function tools **sorted by name** (`_wire_tools`), whatever order the registry handed over.
- **The TTL** is five minutes unless ``model_params.json`` sets
  ``"cache_control": {"type": "ephemeral", "ttl": "1h"}``. That key is the template of every
  breakpoint the harness places, never Anthropic's top-level automatic-caching field. That field
  is a tail breakpoint placed by the API (*"the last cacheable block"*, taking one of the four
  slots, and a 400 when four explicit ones already exist); the adapter places the same tail
  itself, so all four breakpoints are on the request, where they are counted and a test reads them.
- **The proof is the response**, never a declaration: ``usage.cache_read_input_tokens`` rides the
  ``llm`` line as ``cached_tokens=``, and the write beside it as ``cache_write_tokens=`` (writes
  bill at 1.25x for five minutes and 2x for an hour, so the cost needs them). A prefix under the
  model's minimum (512 tokens on the 5.x models) silently does not cache.

The wire translation
--------------------
- **System text.** Only the **leading** system turns become the top-level ``system``; a later one
  (a compaction summary, a step note, the per-wake brief at the tail) is rendered in place as a
  user-role text block marked `SYSTEM_LABEL`, as the Google adapter does. Lifting them to the front
  would put the per-wake brief inside the cached prefix and destroy every hit. (Anthropic's own
  mid-conversation ``role: "system"`` message would carry operator authority, but it may not be
  followed by a user turn — the brief always is — and Claude Sonnet 5 refuses it outright.)
- **Consecutive same-role turns merge into one message**, so a step's tool results, the step note
  after them and an injected image turn travel as one user message with the ``tool_result`` blocks
  first, which is what Anthropic requires of a turn answering parallel calls.
- **The assistant's own blocks go back verbatim, within a wake.** A response carries blocks the
  harness `Message` has no field for — ``thinking`` (passed back unchanged between steps on the 5.x
  models, which keep it), a server tool's ``server_tool_use`` and its result (with the
  ``encrypted_content`` / ``encrypted_index`` the next step needs), citations. The adapter keeps each
  reply's raw blocks for the life of the adapter (one wake) and sends them back in place of the
  text-and-calls rendering. They are **not persisted**: they are opaque vendor state of unbounded
  size, and the transcript's Context Discipline would have to bound a new class for them. A later
  wake replays the conversation as text and calls; earlier thinking is dropped from the front, which
  Anthropic allows. What a later wake does keep of a server tool is its **record**: each
  ``server_tool_use`` block becomes a `ServerToolUse` on the reply (name + what it ran on), which the
  transcript persists and the engine shows beside the turn (`_server_tools`, issue #727), so a page
  this agent fetched yesterday is not a claim it can no longer back.
- **Preserved thinking binds a thinking block to everything sent before it** (the history-editing
  check: enforced for accounts created on or after 2026-08-31, a 400 on a replayed block whose prefix
  changed). The harness does change a prefix inside one wake — a batched wake's second turn no longer
  carries the first turn's brief, and a compaction between turns rewrites what came before. So each
  kept reply records a digest of the exact wire prefix it was produced on (`_Stream`), and a replay
  whose prefix no longer matches sends that reply without its thinking blocks. Every later reply
  produced before that strip then mismatches by itself, which is Anthropic's chain rule; one
  produced after it, on the stripped prefix, still matches and keeps its thinking. Text, calls and server-tool blocks still go back.
- **``pause_turn``.** A server tool's long turn may pause; the adapter resends the paused assistant
  content unchanged and continues, up to `PAUSE_CONTINUATIONS` times, each continuation its own
  billed call with its own ``llm`` line. A turn still paused after that is a transient response
  fault, never a fragment committed as an answer.
- **A server tool and a client tool in one parallel group** come back with ``stop_reason:
  "tool_use"`` and the server tool not yet run; the raw blocks go back with the client results, and
  Anthropic runs it then — but only if the message answering the group holds ``tool_result`` blocks
  and nothing else (measured live, issue #735). The engine's step note, and any other system note
  or injected image after a step's results, is therefore folded into the last result's ``content``
  there (`_fold_into_results`); left as a trailing block it was a 400 on every mixed group. Its
  result therefore arrives in the *next* reply, so a failure there is marked on the record of the
  reply that made the call (`_mark_late_failures`, issue #737). A server tool's own failure arrives inside an HTTP 200 (its result's
  ``content`` is an error object where success is a list) and is logged, never raised.

Every vendor control, and the three the 5.x models refuse
---------------------------------------------------------
``model_params.json`` is passed to ``messages.create`` as written — ``output_config`` (``effort``),
``thinking`` (``display``), ``max_tokens``, ``stop_sequences``, ``inference_geo``, ``service_tier``,
``metadata``, ``extra_body`` and the rest the SDK names — and nothing is sent that it does not set, so
an agent runs Anthropic's defaults by default. Two keys are checked at startup instead of at the
vendor: a key the SDK does not name, and **``temperature`` / ``top_p`` / ``top_k``**, which are a 400
on every 5.x model and an error on every model from API version 2026-10-21 — a config fault the
operator sees once, never a 400 on every wake. ``max_tokens`` is required by the API; unset, the
adapter sends the model's own maximum from the Models API, the nearest thing to *uncapped* this
API has.

Faults, by nature
-----------------
429 and 529 (``overloaded_error``) and the rest of 5xx are transient; a 504 is the server-side twin
of a read timeout; 401/403 are the credential; a ``billing_error`` or a credit-balance refusal is
the billing class; an over-length prompt is the context class; every other 400 propagates. A
``stop_reason: "refusal"`` is Anthropic's final word on the content (the 5.x models have no
server-side fallback on Haiku 5.5): logged with its ``stop_details`` category, never retried.

Cost — computed, never vendor-stated
------------------------------------
Anthropic states no price, so ``cost=`` is computed from its published rates
(`basecradle_harness._anthropic_rates`) and tagged ``cost_basis=computed``. A web search's fee
($10 per 1,000) is a priced ``media`` line (``kind=search.web``), and the ``llm`` line carries the
call's own count as ``web_search_requests=`` beside it — with ``web_fetch_requests=`` and
``code_execution_requests=``, which cost nothing extra and are the journal's only evidence that a
server tool ran (each field rendered only when non-zero). ``provider=anthropic``; no ``endpoint=``, the
vendor *is* the endpoint.

Stateless per turn: the full conversation is sent every call and the harness owns history. This
adapter never streams; every call carries the harness's fitted budget as its ``timeout``, which is
also what keeps the SDK's own ten-minute non-streaming guard out of the way.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import logging
import os
import time
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any

from basecradle_harness._anthropic_rates import (
    CONTAINER_GAP,
    call_cost,
    known,
    server_tool_requests,
    usage_of,
    web_search_cost,
)
from basecradle_harness._caching import EXPLICIT
from basecradle_harness._context import is_context_overflow, request_chars
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
    ProviderTimeoutError,
)
from basecradle_harness._faults import is_out_of_funds, is_too_large
from basecradle_harness._messages import (
    ImageContent,
    Message,
    ServerToolUse,
    ToolCall,
    ToolSpec,
)
from basecradle_harness._observability import (
    generation_id,
    generation_of,
    log_llm_call,
    log_media_call,
    token_counts,
)
from basecradle_harness._openai_wire import format_citations
from basecradle_harness._secret import Sealed
from basecradle_harness._server_tools import mark_failed, record, summarize
from basecradle_harness._timeouts import METADATA_TIMEOUT, call_timeout, output_cap

_log = logging.getLogger("basecradle_harness")

#: This adapter's single surface — the Messages API — declared for the SDK-scoped surface contract
#: (issue #163); ``AI_SDK_SURFACE`` is left unset for it.
SURFACES = ("messages",)
DEFAULT_SURFACE = "messages"
#: The endpoint vendor, as the per-call log line names it (``provider=anthropic``).
PROVIDER = "anthropic"
#: The ``AI_SDK`` value — the PyPI distribution the harness imports, as for every adapter (#158).
SDK = "anthropic"
#: Anthropic's own API host, handed to the SDK whenever ``AI_BASE_URL`` is unset — so the SDK never
#: reads ``ANTHROPIC_BASE_URL`` for itself.
DEFAULT_BASE_URL = "https://api.anthropic.com"

#: The server tools this adapter sends, by the builtin name a tool plugin resolves to → the tool
#: definition on the wire. The latest versions on 2026-10-31: web search and web fetch with dynamic
#: filtering and ``response_inclusion``, and the code-execution tool whose results arrive as
#: ``bash_code_execution_tool_result``. None needs a beta header. Sent in this order, ahead of the
#: function tools, so the cached prefix is the same bytes on every call.
BUILTINS: Mapping[str, Mapping[str, str]] = {
    "web_search": {"type": "web_search_20260318", "name": "web_search"},
    "web_fetch": {"type": "web_fetch_20260318", "name": "web_fetch"},
    "code_execution": {"type": "code_execution_20260521", "name": "code_execution"},
}

#: The sampling parameters every 5.x model refuses with a 400 — refused here, at startup, instead.
SAMPLING_PARAMS = ("temperature", "top_p", "top_k")

#: How many times one turn is resumed after ``stop_reason: "pause_turn"`` before the adapter gives up
#: on it as a transient fault. Each pause is the server's own iteration limit on its tools, so this
#: many is already a pathological turn.
PAUSE_CONTINUATIONS = 8

#: How a system turn that is **not** at the head of the conversation reads to Claude, which takes
#: its system prompt at the top level only. The Google adapter's label, for the same reason.
SYSTEM_LABEL = "[System]\n"

#: The ``kind`` a web search's per-use fee is logged under — the literal OpenAI's fee uses, so the
#: fleet's tool-spend column counts both.
SEARCH_KIND = "search.web"

#: The cache-control TTLs Anthropic offers, and the one the breakpoint gets when none is configured.
CACHE_TTLS = ("5m", "1h")
DEFAULT_CACHE_TTL = "5m"

#: The keys `messages.create` takes that the harness composes itself on every call. Refused with a
#: config fault when a library caller passes one; the config layer warns and drops them from
#: ``model_params.json`` before they get here (`_basecradle._OWNED_ANTHROPIC`).
OWNED_PARAMS = frozenset({"model", "messages", "system", "tools", "stream", "timeout"})

#: Block types that are the model's reasoning — sent back between steps, never shown as its reply,
#: and the ones a changed prefix invalidates.
_THINKING = frozenset({"thinking", "redacted_thinking"})

#: Block types a ``cache_control`` breakpoint may ride. A thinking block cannot carry one.
_MARKABLE = frozenset({"text", "image", "document", "tool_use", "tool_result"})

#: How far back one breakpoint looks for an entry an earlier request wrote: *"The system checks at
#: most 20 positions for each breakpoint, counting the breakpoint itself as the first."* A position is
#: a content block, except that a run of consecutive ``tool_use`` blocks is one position and so is a
#: run of consecutive ``tool_result`` blocks (`_RUNS`). Anthropic's prompt-caching documentation,
#: read 2026-10-31 (issue #726).
LOOKBACK = 20

#: The most ``cache_control`` breakpoints one request may carry; a fifth is a 400.
MAX_BREAKPOINTS = 4

#: How many breakpoints are chained back from the anchor. The four slots are spent as anchor + tail
#: + `CHAINED` (issue #729: the tail goes on every step and the chain gives up the slot), a reach of
#: `LOOKBACK` × (1 + `CHAINED`) = 60 positions behind the anchor.
CHAINED = MAX_BREAKPOINTS - 2

#: The block types whose consecutive runs Anthropic counts as a single lookback position.
_RUNS = frozenset({"tool_use", "tool_result"})

#: The server tools' result blocks, and the server-tool name each answers — for the error log.
_RESULTS = {
    "web_search_tool_result": "web_search",
    "web_fetch_tool_result": "web_fetch",
    "code_execution_tool_result": "code_execution",
    "bash_code_execution_tool_result": "code_execution",
    "text_editor_code_execution_tool_result": "code_execution",
}

#: The result blocks that mean code ran in Anthropic's sandbox — what the container-time gap is
#: stated on (`_anthropic_rates.CONTAINER_GAP`).
_CODE_RESULTS = frozenset(
    {
        "code_execution_tool_result",
        "bash_code_execution_tool_result",
        "text_editor_code_execution_tool_result",
    }
)


class AnthropicConfigError(ValueError):
    """The adapter's configuration is unusable — raised at construction, never mid-wake.

    `reason` is a stable slug a caller can file a fault under (the describer reports it as
    ``config:<reason>``): ``missing_api_key``, ``sampling_params``, ``invalid_model_params`` or
    ``invalid_cache_control``. The message never carries the key.
    """

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def require_anthropic():
    """Import and return the ``anthropic`` SDK, or raise a clear "no LLM, by design" error.

    The core depends on **no** vendor SDK — an ``AI_SDK=anthropic`` agent installs the extra
    (``pip install 'basecradle-harness[anthropic]'``). Without it the harness genuinely cannot reach
    a model, so this fails loud and actionable at provider construction.
    """
    try:
        import anthropic  # lazy: the core must import without the vendor SDK
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised via monkeypatched import
        raise ProviderError(
            "The 'anthropic' SDK is not installed, so the harness has no way to reach a model "
            "(this is by design — the core depends on no vendor SDK). Install the SDK your "
            "agent's AI_SDK names: pip install 'basecradle-harness[anthropic]'."
        ) from exc
    return anthropic


def refused_params(params: Mapping[str, Any], extra_body: Any = None) -> list[str]:
    """The sampling parameters `params` (or its ``extra_body``) sets, sorted — ``[]`` for none.

    The one reader of the refusal, shared by the adapter's construction and ``--resolved-config``
    (`_basecradle.resolved_model_params`), so a box whose every wake would 400 reads red before it
    is ever woken.
    """
    found = {key for key in SAMPLING_PARAMS if key in params}
    if isinstance(extra_body, Mapping):
        found |= {key for key in SAMPLING_PARAMS if key in extra_body}
    return sorted(found)


class AnthropicProvider:
    """A `Provider` backed by Anthropic's official ``anthropic`` SDK and its Messages API.

    Satisfies the `Provider` protocol — the engine cannot tell it from any other adapter — but every
    model call goes through ``anthropic``.

    Args:
        model: The Claude model id (e.g. ``"claude-haiku-5-5"``).
        api_key: The Anthropic API key. Falls back to ``AI_API_KEY``; never to any ``ANTHROPIC_*``
            variable (see the module docstring).
        base_url: An endpoint override (``AI_BASE_URL``); ``None`` is `DEFAULT_BASE_URL`.
        timeout: A **fixed** generation budget in seconds, overriding the fit — for a library caller
            that knows its calls. ``None`` (every deployment) fits each call (`_timeouts`).
        client: An already-built ``anthropic.Anthropic`` (or compatible). The seam tests use; when
            given, no key is required and nothing is built.
        builtin_tools: The server tools to send on every turn, by builtin name (`BUILTINS`). A name
            this adapter does not send is ignored — no plugin resolves one for this provider.
        default_params: ``messages.create`` keywords applied to every call, from the operator's
            ``model_params.json`` — ``max_tokens``, ``output_config``, ``thinking``,
            ``stop_sequences``, ``inference_geo``, ``service_tier``, ``metadata``, ``extra_body``,
            ``extra_headers`` and the rest the SDK names. ``cache_control`` is the template of the
            harness's one breakpoint (its TTL), never the top-level field. Validated here.
    """

    #: Anthropic caches only what the client marks — the explicit mode the engine places one
    #: breakpoint for (`_caching`). Declared, never inferred: forgetting it costs full freight forever.
    cache_mode = EXPLICIT

    #: The ``AI_SDK`` this adapter is, and the one surface it speaks (issue #564).
    sdk = SDK
    surface = DEFAULT_SURFACE

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        client: Any | None = None,
        builtin_tools: Sequence[str] = (),
        **default_params: Any,
    ) -> None:
        self.model = model
        self.provider = PROVIDER
        #: The total prompt Anthropic reported for the most recent call (issue #276): its uncached,
        #: cache-written and cache-read tokens together.
        self.last_tokens_in: int | None = None
        #: Why Anthropic stopped generating on that call, in its own words (issue #490).
        self.last_finish_reason: str | None = None
        #: The generation budget applied to the most recent call, in seconds (issue #589).
        self.last_timeout: float | None = None
        self._timeout_scale = 1.0
        self._fixed_timeout = timeout
        self._anthropic = require_anthropic()
        self._params, self._cache_control = validated_params(default_params, self._anthropic)
        self._builtins = [dict(BUILTINS[name]) for name in BUILTINS if name in set(builtin_tools)]
        #: Each reply this adapter returned, by identity, with its raw blocks and the digest of the
        #: prefix it was produced on — what the next step sends back (see the module docstring).
        self._replies: dict[int, tuple[Message, list[dict[str, Any]], str]] = {}
        #: The Models API record for this model, read once (`_model_info`).
        self._info: Any = None
        #: Models already warned about for having no rate, and gaps already stated, this wake.
        self._unpriced: set[str] = set()
        self._gap_stated = False
        if client is not None:
            self._client = Sealed(client)
            return
        key = (api_key or os.environ.get("AI_API_KEY") or "").strip()
        if not key:
            raise AnthropicConfigError(
                "missing_api_key",
                "No Anthropic API key: pass api_key=... or set AI_API_KEY. The adapter never lets "
                "the SDK look for one of its own (ANTHROPIC_API_KEY, a profile, an auth token).",
            )
        if os.environ.get("ANTHROPIC_CUSTOM_HEADERS"):
            _log.warning(
                "ANTHROPIC_CUSTOM_HEADERS is set in this environment; the anthropic SDK adds those "
                "headers to every request to Anthropic, and they cannot be refused through it."
            )
        #: The endpoint this adapter was built with.
        self.base_url = base_url or DEFAULT_BASE_URL
        self._client = Sealed(
            self._anthropic.Anthropic(
                api_key=key,
                base_url=self.base_url,
                # The engine's bounded retry is the one policy for every adapter (`_retry`); the
                # SDK's own two retries would compose with it and re-send a timed-out request into
                # the same wall.
                max_retries=0,
                open_telemetry=False,
            )
        )

    # --- capabilities ---------------------------------------------------------

    @property
    def tuning(self) -> dict[str, Any]:
        """What this adapter adds to every call — for an agent's brain, its ``model_params.json``."""
        tuned = copy.deepcopy(self._params)
        if self._cache_control is not None:
            tuned["cache_control"] = dict(self._cache_control)
        return tuned

    def bind_timeout_scale(self, scale: float) -> None:
        """How much of its fitted generation budget this adapter's next calls get (issue #589)."""
        self._timeout_scale = max(1.0, float(scale))

    def supports_vision(self) -> bool | None:
        """Whether the model accepts image input, as the Models API states it (issue #228).

        ``capabilities.image_input.supported``; ``None`` when the record cannot be read, which the
        vision gate reads as "show the image".
        """
        support = getattr(getattr(self._info_or_none(), "capabilities", None), "image_input", None)
        supported = getattr(support, "supported", None)
        return supported if isinstance(supported, bool) else None

    def supports_video(self) -> bool | None:
        """``False``: no Claude model takes video input, so a clip reaches it as frames (#471)."""
        return False

    def context_limit(self) -> int | None:
        """The model's context window, from the Models API's ``max_input_tokens`` (issue #276)."""
        limit = getattr(self._info_or_none(), "max_input_tokens", None)
        return limit if isinstance(limit, int) and not isinstance(limit, bool) else None

    # --- the call -------------------------------------------------------------

    def chat(self, messages: Sequence[Message], tools: Sequence[ToolSpec] | None = None) -> Message:
        """Run one model turn through the SDK and return the assistant's reply."""
        request, digest, base = self._request(messages, tools)
        budget = call_timeout(
            request_chars(messages, tools),
            output_tokens=output_cap(request),
            scale=self._timeout_scale,
            fixed=self._fixed_timeout,
        )
        self.last_timeout = budget.generation
        request["timeout"] = self._anthropic.Timeout(**budget.phases())
        blocks: list[dict[str, Any]] = []
        response: Any = None
        for _ in range(PAUSE_CONTINUATIONS + 1):
            if blocks:
                # Resume a paused turn: the paused content goes back unchanged as the assistant's
                # turn, and the response continues it (Anthropic's documented continuation).
                request["messages"] = [*request["messages"][:base], _assistant(blocks)]
            response = self._call(request)
            blocks.extend(_raw(block) for block in response.content or ())
            if response.stop_reason != "pause_turn":
                break
        else:
            raise ProviderResponseError(
                f"Anthropic paused the turn {PAUSE_CONTINUATIONS + 1} times without finishing it; "
                "retrying."
            )
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            _log.warning(
                "Anthropic refused the turn for %s (category=%s): %s The refusal is final; the turn "
                "is not retried.",
                self.model,
                getattr(details, "category", None),
                getattr(details, "explanation", None) or "no explanation given.",
            )
        reply = _reply(blocks)
        self._mark_late_failures(blocks)
        self._replies[id(reply)] = (reply, blocks, digest)
        self._state_code_gap(blocks)
        with generation_of({"id": getattr(response, "id", None)}):
            return reply

    def _mark_late_failures(self, blocks: Sequence[Mapping[str, Any]]) -> None:
        """Mark on its own record a server tool whose failure arrived in this later reply (#737).

        A mixed group's server tool runs only once the client results go back, so its result lands
        in the reply after the one holding its ``server_tool_use``. `_reply` reads a reply alone, so
        the earlier reply — still kept, and the same `Message` the transcript holds — is found here
        by the use's id, at its position among that reply's uses (the order the record keeps).
        """
        for block in blocks:
            content = block.get("content")
            if block.get("type") not in _RESULTS or not isinstance(content, Mapping):
                continue
            if "error_code" not in content:
                continue
            target = block.get("tool_use_id")
            for message, kept, _ in self._replies.values():
                ids = [b.get("id") for b in kept if b.get("type") == "server_tool_use"]
                if target in ids:
                    mark_failed(message, ids.index(target), str(content.get("error_code")))
                    break

    def _call(self, request: dict[str, Any]) -> Any:
        """One billed ``messages.create``, its ``llm`` line, and its search fee's ``media`` line."""
        started = time.monotonic()
        with _mapped_errors(self._anthropic):
            response = self._client.value.messages.create(**request)
        usage = getattr(response, "usage", None)
        wire = _wire_usage(usage)
        self.last_tokens_in = token_counts(wire).get("tokens_in")
        self.last_finish_reason = getattr(response, "stop_reason", None) or None
        searches = server_tool_requests(usage, "web_search_requests")
        written = _count(getattr(usage, "cache_creation_input_tokens", None))
        log_llm_call(
            provider=self.provider,
            model=self.model,
            seconds=time.monotonic() - started,
            usage=wire,
            cost=self._cost(usage),
            finish_reason=self.last_finish_reason,
            extra={
                "cache_write_tokens": written,
                "web_search_requests": searches or None,
                # Free, but the only trace in the journal that a server tool ran at all: none of a
                # server tool's blocks reach the persisted transcript (see the module docstring).
                "web_fetch_requests": server_tool_requests(usage, "web_fetch_requests") or None,
                "code_execution_requests": (
                    server_tool_requests(usage, "code_execution_requests") or None
                ),
            },
            generation_id=generation_id({"id": getattr(response, "id", None)}),
        )
        if searches:
            log_media_call(
                provider=self.provider,
                kind=SEARCH_KIND,
                model=self.model,
                seconds=None,
                count=searches,
                cost=web_search_cost(searches),
            )
        return response

    def _cost(self, usage: Any) -> float | None:
        """This call's dollars at Anthropic's published rates — computed, never vendor-stated."""
        if not known(self.model) and self.model not in self._unpriced:
            self._unpriced.add(self.model)
            _log.warning(
                "No published rate for Anthropic model %r in the harness's table, so its calls "
                "carry no cost= on the llm line. Add the model's row to _anthropic_rates.RATES from "
                "Anthropic's pricing page.",
                self.model,
            )
        ttl = (self._cache_control or {}).get("ttl") or DEFAULT_CACHE_TTL
        return call_cost(
            self.model,
            usage_of(usage, written_ttl=ttl),
            # Where the call actually ran, as Anthropic reports it (``global`` or ``us``, measured
            # live) — so a geography set through extra_body or a workspace default is priced too.
            # A response that reports none falls back to what the request asked for.
            geo=getattr(usage, "inference_geo", None) or self._params.get("inference_geo"),
            tier=getattr(usage, "service_tier", None),
            speed=getattr(usage, "speed", None),
        )

    def _state_code_gap(self, blocks: Sequence[Mapping[str, Any]]) -> None:
        """One WARNING per wake when code ran on a request whose container time Anthropic bills."""
        if self._gap_stated or not any(block.get("type") in _CODE_RESULTS for block in blocks):
            return
        if any(tool["name"] in ("web_search", "web_fetch") for tool in self._builtins):
            return  # free beside either: "no additional charges for code execution tool calls"
        self._gap_stated = True
        _log.warning(CONTAINER_GAP)

    def _info_or_none(self) -> Any:
        """The Models API record, or ``None`` when it cannot be read — the fail-safe reads' form."""
        try:
            return self._model_info()
        except ProviderError as exc:
            _log.warning("Could not read the Models API record for %s: %s", self.model, exc)
            return None

    def _model_info(self) -> Any:
        """``models.retrieve(model)``, read once for the life of the adapter, on a flat timeout."""
        if self._info is None:
            with _mapped_errors(self._anthropic):
                self._info = self._client.value.models.retrieve(
                    self.model, timeout=METADATA_TIMEOUT
                )
        return self._info

    # --- harness -> Anthropic -------------------------------------------------

    def _request(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec] | None
    ) -> tuple[dict[str, Any], str, int]:
        """This turn's ``messages.create`` keywords, the digest of the prefix, and its length."""
        request: dict[str, Any] = copy.deepcopy(self._params)
        if "max_tokens" not in request:
            ceiling = getattr(self._model_info(), "max_tokens", None)
            if not isinstance(ceiling, int) or isinstance(ceiling, bool) or ceiling <= 0:
                raise ProviderError(
                    f"The Models API states no max_tokens for {self.model}, and the Messages API "
                    "requires one: set max_tokens in model_params.json."
                )
            request["max_tokens"] = ceiling
        wire_tools = _wire_tools(self._builtins, tools)
        system, wire, stream = self._to_wire(messages, wire_tools)
        request["model"] = self.model
        request["messages"] = wire
        if system:
            request["system"] = system
        if wire_tools:
            request["tools"] = wire_tools
        return request, stream.digest(), len(wire)

    def _to_wire(
        self, messages: Sequence[Message], tools: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], _Stream]:
        """The conversation as ``(system blocks, messages, prefix digest)``, merged by role.

        Every block is fed to a `_Stream` digest in send order (without its ``cache_control``,
        which Anthropic's check ignores); a kept reply is sent back raw only while the digest at its
        position still equals the one it was produced on. The anchored turn is marked.

        Kept replies the conversation no longer holds are forgotten here, so the memo is bounded by
        the conversation in flight rather than by the adapter's life: a forgotten reply is rendered
        as text and calls, which is what every later wake sends anyway.
        """
        index = 0
        system: list[dict[str, Any]] = []
        while index < len(messages) and messages[index].role == "system":
            message = messages[index]
            if message.content:
                system.append({"type": "text", "text": message.content})
                if message.cache_anchor:
                    self._mark(system)
            index += 1
        stream = _Stream(system, tools)
        wire: list[dict[str, Any]] = []
        present: set[int] = set()
        for message in messages[index:]:
            role, blocks = "user", []
            if message.role == "assistant":
                role = "assistant"
                kept = self._replies.get(id(message))
                if kept is not None and kept[0] is message:
                    present.add(id(message))
                    # Each kept reply is judged on its own digest. No latch is needed for Anthropic's
                    # chain rule: the digest covers the stream *as sent*, so once an earlier reply
                    # goes back stripped, every later reply produced before that strip mismatches by
                    # itself — and one produced after it (on the stripped prefix) still matches, and
                    # keeps the thinking its own tool round needs.
                    blocks = _replayed(kept[1], message, strip=stream.digest() != kept[2])
                else:
                    blocks = _rendered(message)
            elif message.role == "tool":
                blocks = [
                    {
                        "type": "tool_result",
                        "tool_use_id": message.tool_call_id or "",
                        "content": message.content or "",
                    }
                ]
            elif message.role == "system":
                if message.content:
                    blocks = [{"type": "text", "text": SYSTEM_LABEL + message.content}]
            else:
                if message.content:
                    blocks.append({"type": "text", "text": message.content})
                blocks.extend(_image_block(image) for image in message.images)
                if message.videos:
                    raise ProviderError(
                        "Claude takes no video input, and a clip reached the Anthropic adapter "
                        "unconverted; it is never dropped silently."
                    )
            if not blocks:
                continue
            for block in blocks:
                stream.add(role, block)
            if wire and wire[-1]["role"] == role:
                wire[-1]["content"].extend(blocks)
            else:
                wire.append({"role": role, "content": list(blocks)})
            if message.cache_anchor:
                self._mark(blocks)
        self._replies = {key: kept for key, kept in self._replies.items() if key in present}
        for before, after in pairwise(wire):
            if before["role"] == "assistant" and _pending_server_tool(before["content"]):
                _fold_into_results(after["content"])
        sent = [*system, *(block for turn in wire for block in turn["content"])]
        if tools and not any("cache_control" in block for block in sent):
            # No frozen turn to anchor (a first wake: the charter rides the brief, so the frozen
            # transcript is empty). The stable/volatile boundary is then the end of the tool list,
            # which renders first and is the same bytes on every call of this agent.
            tools[-1]["cache_control"] = self._template()
        # The tool list renders before the first system or message block, so its end is the
        # position just before them: an entry a first wake wrote there is still worth reaching.
        self._place_breakpoints(sent, first=-1 if tools else 0, tail=self._mark_tail(sent))
        return system, wire, stream

    def _template(self) -> dict[str, Any]:
        """A fresh breakpoint: the operator's ``cache_control`` (its TTL), else the 5-minute one."""
        return dict(self._cache_control or {"type": "ephemeral"})

    def _mark(self, blocks: list[dict[str, Any]]) -> None:
        """Put the breakpoint on the last block that can carry one, in place (a fresh dict)."""
        for position in range(len(blocks) - 1, -1, -1):
            if _markable(blocks[position]):
                blocks[position]["cache_control"] = self._template()
                return

    def _mark_tail(self, blocks: list[dict[str, Any]]) -> int | None:
        """Put the tail breakpoint on the request's last block that can carry one (issue #729).

        Within one wake every step re-sends the step before it and adds its own growth, so step *k*'s
        tail entry is the prefix step *k + 1* reads: the per-wake brief, the new turn and the wake's
        own tool chain are written once and read on every later step, rather than billed in full on
        each. Returns the block's index, or ``None`` when no block after the anchor can carry one
        (the walk stops at an existing breakpoint rather than marking behind it).
        """
        for position in range(len(blocks) - 1, -1, -1):
            if "cache_control" in blocks[position]:
                return None
            if _markable(blocks[position]):
                blocks[position]["cache_control"] = self._template()
                return position
        return None

    def _place_breakpoints(
        self, blocks: list[dict[str, Any]], *, first: int = 0, tail: int | None = None
    ) -> None:
        """Walk back from the anchor's breakpoint, adding one every `LOOKBACK` positions (#726).

        `blocks` is the request's system blocks and then every message block, in send order; `first`
        is the earliest position an entry can sit at (-1 when a tool list precedes them); `tail` is
        the index of the tail breakpoint `_mark_tail` placed, if any. The four slots are spent as
        **anchor + tail + two chained** (`CHAINED`), and the tail's slot is reserved on every
        request, whether or not a tail could be placed: one fixed split, verifiable by eye.

        The anchor's breakpoint finds an entry only within `LOOKBACK` positions of itself, and the
        entry a wake can find is the one the *previous* wake wrote at its own anchor — the end of the
        transcript as it stood then. So a previous wake that added `LOOKBACK` or more positions of
        work (about seven sequential tool steps: a call, its result and a step note each) leaves this
        wake's anchor blind to it, and the whole prefix is written again at the write rate: more than
        sending no breakpoint at all.

        Each added breakpoint starts a lookback window of its own, so they are chained back from the
        anchor, each on the earliest position the window before it still reaches. Spaced exactly
        `LOOKBACK` apart the windows tile with no gap, and the reach is `LOOKBACK` × (1 + `CHAINED`)
        positions behind the anchor (60). A breakpoint that hits is a read; the write is charged once,
        for the span after the hit, so the extra breakpoints cost nothing. The chain is walked from
        the anchor only, never from the tail: what lies behind the tail and after the anchor is this
        wake's own work, which the previous step's tail entry already covers.

        Nothing is added when the earliest breakpoint's own window already reaches `first`, and the
        request never carries more than the vendor's four: a caller that anchored more than the three
        the tail leaves keeps its latest three, with a WARNING, rather than a 400. Each lands on the
        **last** block of its position (a run of parallel calls or results is one position), and only
        on a block that can carry one; where the earliest reachable position cannot, the next later
        one is taken, which narrows the reach and never opens a gap. Every block is counted as a
        position except the runs Anthropic names, so an uncertain count can only overstate a
        distance, which also never opens a gap.
        """
        anchors = MAX_BREAKPOINTS - 1  # the tail's slot is reserved on every request
        marked = [
            index
            for index, block in enumerate(blocks)
            if "cache_control" in block and index != tail
        ]
        if not marked:
            return
        if len(marked) > anchors:
            _log.warning(
                "%d turns were anchored for caching; Anthropic takes at most %d breakpoints and one "
                "is reserved for the request's tail, so the latest %d are kept.",
                len(marked),
                MAX_BREAKPOINTS,
                anchors,
            )
            for index in marked[:-anchors]:
                del blocks[index]["cache_control"]
            marked = marked[-anchors:]
        positions = _positions(blocks)
        earliest = marked[0]
        for _ in range(anchors - len(marked)):
            reach = positions[earliest] - LOOKBACK
            if reach < first:
                return
            candidate = next(
                (
                    index
                    for index in range(earliest)
                    if positions[index] >= reach
                    and _markable(blocks[index])
                    and positions[index + 1] != positions[index]
                ),
                None,
            )
            if candidate is None:
                return
            blocks[candidate]["cache_control"] = self._template()
            earliest = candidate

    # --- lifecycle --------------------------------------------------------------

    def close(self) -> None:
        """Close the SDK client this adapter built."""
        closer = getattr(self._client.value, "close", None)
        if callable(closer):
            closer()

    def __enter__(self) -> AnthropicProvider:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# --- helpers --------------------------------------------------------------------


class _Stream:
    """A running digest of a request's prefix: its system blocks, its tools, then each block in order.

    A digest of the *stream* of ``(role, block)`` pairs rather than of the merged JSON, so it is the
    same whether two turns of one role were sent as one message or two (Anthropic combines them),
    and it can be read at any position without re-serializing what came before.
    """

    def __init__(self, system: list[dict[str, Any]], tools: list[dict[str, Any]]) -> None:
        self._hash = hashlib.sha256()
        self._feed(
            {
                "system": [_unmarked(block) for block in system],
                "tools": [_unmarked(tool) for tool in tools],
            }
        )

    def add(self, role: str, block: Mapping[str, Any]) -> None:
        self._feed({"role": role, "block": _unmarked(block)})

    def digest(self) -> str:
        return self._hash.copy().hexdigest()

    def _feed(self, item: Any) -> None:
        text = json.dumps(item, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        self._hash.update(text.encode("utf-8"))
        self._hash.update(b"\x1e")


def _unmarked(block: Mapping[str, Any]) -> dict[str, Any]:
    """`block` without its ``cache_control`` — a breakpoint never changes what a prefix *is*."""
    return {key: value for key, value in block.items() if key != "cache_control"}


def _markable(block: Mapping[str, Any]) -> bool:
    """Whether `block` can carry a breakpoint: a markable type, and not empty (*"Empty text blocks
    cannot be cached"*; an empty tool result is held to the same rule)."""
    kind = block.get("type")
    if kind not in _MARKABLE:
        return False
    if kind == "text":
        return bool(block.get("text"))
    if kind == "tool_result":
        return bool(block.get("content"))
    return True


def _positions(blocks: Sequence[Mapping[str, Any]]) -> list[int]:
    """Each block's lookback position: one per block, a run of `_RUNS` blocks of one type sharing one."""
    positions: list[int] = []
    position, previous = -1, None
    for block in blocks:
        kind = block.get("type")
        if kind not in _RUNS or kind != previous:
            position += 1
        positions.append(position)
        previous = kind
    return positions


def validated_params(
    params: Mapping[str, Any], anthropic: Any = None
) -> tuple[dict[str, Any], Any]:
    """`params` checked against what ``messages.create`` takes: ``(params, breakpoint template)``.

    A key the SDK does not name, a harness-owned key (top level or inside ``extra_body``, where the
    SDK would let it override the harness's wiring), a sampling parameter, or an unusable
    ``cache_control`` is an `AnthropicConfigError` here, at startup, rather than an SDK
    ``TypeError`` or a vendor 400 inside a wake. The one reader of all four, shared by the adapter
    and ``--resolved-config`` (`_basecradle.resolved_model_params`). `anthropic` is the SDK module;
    ``None`` imports it, and only the unknown-key check — the one that needs the SDK's signature —
    is skipped where it is not installed.
    """
    params = dict(params)
    extra_body = params.get("extra_body")
    owned = (
        sorted(set(extra_body) & (OWNED_PARAMS | {"cache_control"}))
        if isinstance(extra_body, Mapping)
        else []
    )
    if owned:
        raise AnthropicConfigError(
            "invalid_model_params",
            f"model_params.json's extra_body sets {', '.join(owned)}, which the harness composes "
            "itself on every call; in extra_body it would override that wiring.",
        )
    cache_control = params.pop("cache_control", None)
    refused = refused_params(params, params.get("extra_body"))
    if refused:
        raise AnthropicConfigError(
            "sampling_params",
            f"model_params.json sets {', '.join(refused)}, which every Claude 5.x model refuses "
            "with a 400 (and every model does from API version 2026-10-21). Remove them; tune the "
            "model with output_config.effort instead.",
        )
    if anthropic is None:
        try:
            anthropic = require_anthropic()
        except ProviderError:
            anthropic = None
    unknown: list[str] = []
    if anthropic is not None:
        accepted = set(inspect.signature(anthropic.resources.messages.Messages.create).parameters)
        unknown = sorted(set(params) - (accepted - OWNED_PARAMS - {"self"}))
    if unknown:
        raise AnthropicConfigError(
            "invalid_model_params",
            f"model_params.json sets {', '.join(unknown)}, which the anthropic SDK's "
            "messages.create does not take (or the harness sets itself).",
        )
    if cache_control is not None:
        ttl = cache_control.get("ttl") if isinstance(cache_control, Mapping) else None
        if (
            not isinstance(cache_control, Mapping)
            or cache_control.get("type") != "ephemeral"
            or set(cache_control) - {"type", "ttl"}
            or (ttl is not None and ttl not in CACHE_TTLS)
        ):
            raise AnthropicConfigError(
                "invalid_cache_control",
                'model_params.json\'s cache_control must be {"type": "ephemeral"} with an optional '
                f"ttl of {' or '.join(repr(t) for t in CACHE_TTLS)}.",
            )
        cache_control = dict(cache_control)
    return params, cache_control


def _wire_tools(builtins: list[dict[str, Any]], tools: Sequence[ToolSpec] | None) -> list[dict]:
    """The server tools in `BUILTINS` order, then the function tools sorted by name — byte-stable."""
    declared = [
        {
            "name": spec.name,
            "description": spec.description,
            "input_schema": spec.parameters or {"type": "object", "properties": {}},
        }
        for spec in sorted(tools or (), key=lambda spec: spec.name)
    ]
    return [dict(tool) for tool in builtins] + declared


def _rendered(message: Message) -> list[dict[str, Any]]:
    """An assistant turn with no kept blocks (an earlier wake's) as text and ``tool_use`` blocks."""
    blocks: list[dict[str, Any]] = []
    if message.content:
        blocks.append({"type": "text", "text": message.content})
    blocks.extend(
        {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
        for call in message.tool_calls
    )
    return blocks


def _replayed(
    raw: Sequence[Mapping[str, Any]], message: Message, *, strip: bool
) -> list[dict[str, Any]]:
    """A kept reply's raw blocks, to send back — its thinking dropped when `strip`.

    Each ``tool_use`` block carries the id of the harness's call in the same position, because the
    ``tool_result`` answering it carries that id (`_engine._unique_call_ids` may have renamed a
    duplicate). A reply whose calls no longer line up with its blocks is rendered afresh instead.
    """
    calls = iter(message.tool_calls)
    blocks: list[dict[str, Any]] = []
    for block in raw:
        if strip and block.get("type") in _THINKING:
            continue
        if block.get("type") == "text" and not block.get("text"):
            continue  # an empty text block is a 400 on the way back in
        block = copy.deepcopy(dict(block))
        if block.get("type") == "tool_use":
            call = next(calls, None)
            if call is None:
                return _rendered(message)
            block["id"] = call.id
        blocks.append(block)
    if next(calls, None) is not None:
        return _rendered(message)
    return blocks


def _pending_server_tool(blocks: Sequence[Mapping[str, Any]]) -> bool:
    """Does this assistant content carry a ``server_tool_use`` with no result yet (issue #735)?

    That is the mixed parallel group: the model called a server tool and a client tool in one
    response, and Anthropic stopped for the client's results before running the server tool.
    """
    answered = {block.get("tool_use_id") for block in blocks if block.get("type") in _RESULTS}
    return any(
        block.get("type") == "server_tool_use" and block.get("id") not in answered
        for block in blocks
    )


def _fold_into_results(blocks: list[dict[str, Any]]) -> None:
    """Make the user message after a pending server tool hold ``tool_result`` blocks only (#735).

    Anthropic runs a pending server tool only when the message answering its group is nothing but
    tool results: measured live on 2026-10-31, ``[tool_result]`` and ``[tool_result]`` with the text
    folded into its ``content`` are accepted, while ``[tool_result, text]`` is a 400 naming the server
    tool's missing result. The engine's step note (and any other system note, or an injected image)
    follows a step's results as such a trailing block, so every mixed group failed on its next step.
    The trailing blocks are moved, in order, into the last result's ``content``; a breakpoint one of
    them carried moves to that result, which is where the position it marked now ends.
    """
    results = [block for block in blocks if block.get("type") == "tool_result"]
    rest = [block for block in blocks if block.get("type") != "tool_result"]
    if not results or not rest:
        return
    last = results[-1]
    content = last.get("content")
    folded: list[dict[str, Any]] = (
        [{"type": "text", "text": content}] if isinstance(content, str) and content else []
    )
    if isinstance(content, list):
        folded = list(content)
    marked = None
    for block in rest:
        moved = dict(block)
        marked = moved.pop("cache_control", None) or marked
        folded.append(moved)
    last["content"] = folded
    if marked is not None:
        last["cache_control"] = marked
    blocks[:] = results


def _assistant(blocks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """A paused turn's content as the assistant message that resumes it — minus empty text blocks,
    which are a 400 on the way back in (as in `_replayed`)."""
    kept = [dict(b) for b in blocks if not (b.get("type") == "text" and not b.get("text"))]
    return {"role": "assistant", "content": kept}


def _image_block(image: ImageContent) -> dict[str, Any]:
    """An image as a Claude content block — inline base64 for a data URL, a URL source otherwise."""
    url = image.url
    if url.startswith("data:"):
        header, _, payload = url.partition(",")
        mime = header[len("data:") :].split(";", 1)[0]
        if not mime or ";base64" not in header:
            raise ProviderError(f"Unsupported inline image URL for Anthropic: {header[:64]!r}.")
        return {"type": "image", "source": {"type": "base64", "media_type": mime, "data": payload}}
    return {"type": "image", "source": {"type": "url", "url": url}}


def _raw(block: Any) -> dict[str, Any]:
    """A response content block as the JSON it arrived as — what is sent back on a later step."""
    if isinstance(block, Mapping):
        return dict(block)
    return block.to_dict(mode="json")


def _reply(blocks: Sequence[Mapping[str, Any]]) -> Message:
    """A turn's blocks as a harness assistant `Message`: its text, its client calls, its sources.

    Thinking is the model's reasoning, not its reply, and is left out; a server tool's blocks are
    the vendor's to run and stay in the kept blocks only — but each ``server_tool_use`` is reported
    on the reply as a `ServerToolUse` (its name and what it was run on), the record a later wake
    reads instead of the blocks it will never see (issue #727). A server tool's failure — which
    Anthropic reports inside the 200 as an error object where a success is a list — is logged.
    """
    texts: list[str] = []
    calls: list[ToolCall] = []
    citations: list[dict[str, str]] = []
    used: list[ServerToolUse] = []
    by_id: dict[str, ServerToolUse] = {}
    for block in blocks:
        kind = block.get("type")
        if kind == "server_tool_use":
            use = ServerToolUse(name=block.get("name") or "", input=summarize(block.get("input")))
            used.append(use)
            by_id[block.get("id") or ""] = use
        elif kind == "text":
            texts.append(block.get("text") or "")
            for cited in block.get("citations") or ():
                if isinstance(cited, Mapping) and cited.get("url"):
                    citations.append({"url": cited["url"], "title": cited.get("title") or ""})
        elif kind == "tool_use":
            arguments = block.get("input")
            calls.append(
                ToolCall(
                    id=block.get("id") or "",
                    name=block.get("name") or "",
                    arguments=dict(arguments) if isinstance(arguments, Mapping) else {},
                )
            )
        elif kind in _RESULTS:
            content = block.get("content")
            if isinstance(content, Mapping) and "error_code" in content:
                failed = by_id.get(block.get("tool_use_id") or "")
                if failed is not None:
                    # The record says so: a fetch that never read the page is not evidence it did.
                    failed.failed = str(content.get("error_code") or "failed")
                _log.warning(
                    "Anthropic's %s server tool failed inside the turn (error_code=%s); the model "
                    "was told and carried on.",
                    _RESULTS[kind],
                    content.get("error_code"),
                )
    text = "".join(texts)
    if text and citations:
        text += format_citations(citations)
    reply = Message.assistant(content=text or None, tool_calls=calls)
    reply.server_tools = record(used)
    return reply


def _wire_usage(usage: Any) -> dict[str, Any] | None:
    """Anthropic's ``usage`` in the harness's usage vocabulary (the chat-wire spelling).

    One translation, at the vendor boundary, so the shared readers need no Anthropic branch:
    ``tokens_in`` is the whole prompt — Anthropic reports it in three disjoint parts, uncached,
    cache-written and cache-read — and ``cached_tokens`` is the read part, the witness that caching
    is doing anything. ``tokens_out`` includes the thinking, which is billed as output.
    """
    if usage is None:
        return None
    uncached = _count(getattr(usage, "input_tokens", None))
    output = _count(getattr(usage, "output_tokens", None))
    if uncached is None and output is None:
        return None
    written = _count(getattr(usage, "cache_creation_input_tokens", None)) or 0
    read = _count(getattr(usage, "cache_read_input_tokens", None))
    prompt = (uncached or 0) + written + (read or 0)
    wire: dict[str, Any] = {
        "prompt_tokens": prompt,
        "completion_tokens": output or 0,
        "total_tokens": prompt + (output or 0),
    }
    if read is not None:
        wire["prompt_tokens_details"] = {"cached_tokens": read}
    return wire


def _count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


class _mapped_errors:
    """Map ``anthropic`` failures onto the harness's typed errors — by nature, never by vendor.

    A context overflow compacts and retries, an oversized request is reported once, an account out
    of credit is the billing class, a 429 and a 529 and the rest of 5xx are transient, a 504 is a
    server-side timeout, and a transport failure or timeout is transient. A generic 400 propagates
    as a plain `ProviderAPIError`, so a fixable config defect never loses the peer's message.
    """

    def __init__(self, anthropic: Any) -> None:
        self._anthropic = anthropic

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is None:
            return False
        sdk = self._anthropic
        if isinstance(exc, sdk.APIStatusError):
            raise _from_status_error(exc) from exc
        if isinstance(exc, sdk.APITimeoutError):
            raise ProviderTimeoutError(f"Anthropic did not answer in time: {exc}") from exc
        if isinstance(exc, sdk.APIConnectionError):
            raise ProviderConnectionError(f"Could not reach Anthropic: {exc}") from exc
        if isinstance(exc, sdk.APIResponseValidationError):
            raise ProviderResponseError(
                f"Anthropic returned an unparseable response: {exc}"
            ) from exc
        if isinstance(exc, (TypeError, ValueError)):
            # The SDK refusing to build the request — a configuration fault, so a plain provider
            # error that propagates and leaves the peer's message re-drivable.
            raise ProviderError(f"The anthropic SDK refused the request: {exc}") from exc
        return False


def _from_status_error(exc: Any) -> ProviderError:
    """An ``anthropic.APIStatusError`` as the right typed `ProviderError`."""
    status = getattr(exc, "status_code", None) or 0
    body = getattr(exc, "body", None)
    error = body.get("error") if isinstance(body, Mapping) else None
    kind = error.get("type") if isinstance(error, Mapping) else None
    message = (error.get("message") if isinstance(error, Mapping) else None) or str(exc)
    text = json.dumps(body) if isinstance(body, (Mapping, list)) else str(body or "")
    retry_after = _retry_after(getattr(exc, "response", None))
    if status in (400, 413) and is_context_overflow(message):
        return ProviderContextLengthError(message, status_code=status, body=text)
    if status == 413 or kind == "request_too_large" or (status == 400 and is_too_large(message)):
        return ProviderPayloadTooLargeError(message, status_code=status, body=text)
    if kind == "billing_error" or status == 402 or is_out_of_funds(message):
        return ProviderBillingError(message, status_code=status, body=text)
    if status in (401, 403):
        return ProviderAuthError(
            f"Anthropic refused the call (HTTP {status}): {message}", status_code=status, body=text
        )
    if status == 429:
        return ProviderRateLimitError(
            f"Anthropic rate-limited the request (HTTP 429): {message}",
            status_code=status,
            body=text,
            retry_after=retry_after,
        )
    if status == 504:
        return ProviderTimeoutError(f"Anthropic's deadline ran out (HTTP 504): {message}")
    if status >= 500:
        return ProviderServerError(
            f"Anthropic failed on its own side (HTTP {status}{f', {kind}' if kind else ''}).",
            status_code=status,
            body=text,
            retry_after=retry_after,
        )
    return ProviderAPIError(message, status_code=status, body=text)


def _retry_after(response: Any) -> float | None:
    headers = getattr(response, "headers", None)
    raw = headers.get("retry-after") if headers is not None else None
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None
