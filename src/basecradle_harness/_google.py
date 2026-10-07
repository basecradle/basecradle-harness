"""The native Google adapter — Gemini on Vertex AI over the official ``google-genai`` SDK (issue #655).

The fourth `Provider` adapter. It reaches Gemini through Google's own first-party SDK
(``google-genai`` on PyPI, imported as ``google.genai``) in **Vertex AI mode**: a Google Cloud
project, a location, and a service-account credential — no API key, no OpenAI-compatibility shim, no
harness-owned HTTP. Selected by ``AI_PROVIDER=google`` + ``AI_SDK=google-genai``.

Why direct, and why Vertex
--------------------------
The fleet calls first-party models it holds an account with **direct**, and keeps OpenRouter for
open-weight models with many hosts. Before this adapter the only road to Gemini was OpenRouter, which
added a party and a fee to the data path and, worse, could not reach Google's newest Flash models
inside the United States at all: Google serves them from ``global`` or the ``us`` multi-region, and
OpenRouter's US host has no record for them. Vertex at location ``us`` is Google's own guarantee that
processing stays in the United States, so it is the stronger residency claim, not just a cheaper one.

The credential, and what it never falls back to
-----------------------------------------------
A service-account **JSON file**, named by path (`AI_CREDENTIALS_FILE`; a relative path is read
against the agent's config home). Never the JSON in an environment variable — the NOC places secrets
as files, with file permissions — and never Application Default Credentials: the adapter always
hands the SDK the credentials it loaded, because the SDK's own fallback (``google.auth.default()``)
would pick up whatever ``gcloud`` login happens to be on the box, which is exactly the silent
borrowing of someone else's account that this fleet's auth fences exist to stop. Only a
``service_account`` file loads; a user's ``authorized_user`` file is refused.

**The token is refreshed by the adapter, bounded.** Left to the SDK, the service-account token is
refreshed through ``google-auth``'s defaults — 120 seconds a request and up to three attempts with
backoff — a wait no fitted budget sees and a retry under the engine's own. So the adapter refreshes
it first, through a one-shot transport with the fixed connect budget (`_token_request`), and the
SDK finds it valid.

**The location has no default.** The SDK defaults an unset location to ``global``, which lets Google
process the call anywhere; for an operator who chose this provider *for* its residency guarantee that
default is the one failure they would never see. So an unset `AI_LOCATION` is a configuration fault,
loud at startup. The project defaults to the one the credential file itself names.

Single native surface
---------------------
``generateContent`` is the one surface this adapter speaks (`SURFACES`), so ``AI_SDK_SURFACE`` is
left unset for it, exactly as for the native xAI and OpenRouter adapters.

The wire translation, and the three places Gemini differs
---------------------------------------------------------
- **System text.** Gemini takes a ``system_instruction`` and has no system *role* inside the
  conversation, while the harness places system turns mid-transcript (a compaction summary, a step
  note, and — at the tail — the per-wake brief). Only the **leading** system turns become the
  ``system_instruction``; a later one is rendered in place as a user-role turn marked
  `SYSTEM_LABEL`. Lifting them all to the front would put the per-wake brief at position zero, which
  changes the cacheable prefix on every request and silently destroys implicit caching
  (`CLAUDE.md` → Context Discipline, the cache invariant).
- **Tool results** answer a call **by name**, not only by id, and a step's results travel together:
  consecutive ``tool`` turns become one user-role ``Content`` of ``functionResponse`` parts, in call
  order (Google: interleaving a parallel batch's calls and responses is a 400).
- **Thought signatures** (Gemini 3). A model that calls a function returns an opaque signature with
  the call, and the current turn's calls must carry theirs back or the request is refused with a 400.
  The adapter keeps every signature it is handed for the life of the adapter (one wake) and sends it
  back on its call — on every call it holds one for, because the engine appends a step note before
  each model call and Google counts that note as a new turn, so "the current turn" is usually empty
  by the time a request is built. Signatures are **not** persisted to the transcript: they are opaque vendor state
  of unbounded size, and the transcript's Context Discipline would have to bound a new class for
  them. So a current-turn call whose signature did not survive — a turn resumed after a crash, the
  one case where the current turn was started by an earlier process — carries Google's documented
  bypass value (`SKIP_SIGNATURE`) instead, which Google says costs reasoning quality and never
  correctness. Signatures on text parts are optional per Google and are not kept.

Cost — computed, never vendor-stated
------------------------------------
Vertex reports tokens and no dollars, so ``cost=`` on this adapter's ``llm`` line is **computed**
from Google's published rates (`basecradle_harness._google_rates`), by the capital's ruling on #655 —
one of the two vendors (OpenAI is the other, #657) whose cost is harness arithmetic, tagged
``cost_basis=computed`` on the line. A model, a location class, or a traffic tier the
table does not carry gets **no** ``cost=``, and one WARNING per model per adapter, never a guess.

Server-side built-ins, and the one Vertex will not combine (issue #656)
----------------------------------------------------------------------
Two of Gemini's built-ins ride **beside** the harness's function declarations on every turn, opted in
like every provider's powerful built-ins: **code execution** (Python in Google's sandbox) and **URL
context** (the model reads up to 20 URLs itself). Both bill as tokens only — the code, its result
and the fetched pages arrive as ``tool_use_prompt_token_count``, priced at the input rate — so they
need nothing beyond the call's own ``cost=``.

**Google Search grounding cannot ride beside them.** Vertex: *"The Gemini API doesn't support
combining search tools (such as googleSearch) with non-search tools (such as function calling …) in
the same generateContent request"*, and every harness turn carries function declarations. So Search
is a harness-run tool instead (`basecradle_harness._google_search`): the agent calls ``web_search``,
and the harness makes **one grounded call** (`GoogleProvider.search`) with ``google_search`` as its
only tool — the combination Vertex does accept — and hands back the answer with a ``Sources:``
footer built from the grounding metadata. That call writes its own ``llm`` line
(``purpose=helper kind=search.grounding``), and its grounding fee, which is not tokens, its own
priced ``media`` line (`_google_rates.grounding_cost`).

Stateless per turn: the full conversation is sent every call and the harness owns history. This
adapter never streams.
"""

from __future__ import annotations

import base64
import copy
import json
import logging
import mimetypes
import os
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx

from basecradle_harness._caching import AUTOMATIC
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
from basecradle_harness._google_rates import (
    call_cost,
    from_usage_metadata,
    grounding_cost,
    grounding_units,
    known,
)
from basecradle_harness._messages import ImageContent, Message, ToolCall, ToolSpec, VideoContent
from basecradle_harness._observability import (
    HELPER,
    finish_reason,
    generation_id,
    generation_of,
    log_llm_call,
    log_media_call,
    token_counts,
)
from basecradle_harness._openai_wire import format_citations
from basecradle_harness._timeouts import (
    METADATA_TIMEOUT,
    CallTimeout,
    call_timeout,
    output_cap,
)

_log = logging.getLogger("basecradle_harness")

#: This adapter's single surface — Gemini's ``generateContent`` — declared for the SDK-scoped surface
#: contract (issue #163); ``AI_SDK_SURFACE`` is left unset for it.
SURFACES = ("native",)
DEFAULT_SURFACE = "native"
#: The endpoint vendor, as the per-call log line names it (``provider=google``).
PROVIDER = "google"
#: The ``AI_SDK`` value — the PyPI distribution the harness imports, as for every adapter (#158).
SDK = "google-genai"

#: The server-side built-ins this adapter sends beside the function declarations, by the builtin name
#: a tool plugin resolves to → the ``types.Tool`` field that enables it. Google Search is not here:
#: Vertex refuses it beside function calling (see the module docstring), so it is `search` instead.
BUILTINS = {"code_execution": "code_execution", "url_context": "url_context"}

#: The ``kind`` the grounded search call and its fee are logged under — one name for the ``llm``
#: line and the ``media`` line, so the two halves of one search are one grep apart.
SEARCH_KIND = "search.grounding"

#: The variables the brain's Vertex configuration is read from, when the caller passes none.
CREDENTIALS_FILE_VAR = "AI_CREDENTIALS_FILE"
LOCATION_VAR = "AI_LOCATION"
PROJECT_VAR = "AI_PROJECT"

#: The OAuth scope a Vertex call needs — the one Google's own SDK asks for.
CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"

#: Google's documented stand-in for a function call's thought signature, for a call whose real one
#: is not available ("transferring history from a model that does not include thought signatures").
#: Google: "this should be a last resort as it will negatively impact model performance". It is
#: bytes here because the SDK base64-encodes the field on the wire, which is the form Google expects.
SKIP_SIGNATURE = b"skip_thought_signature_validator"

#: The prefix of a call id the **adapter** made up, because the model returned none. Such an id is
#: never sent back to Google: the model never issued it, and the harness only needs it to pair a
#: call with its result inside the transcript. An id the model did issue is echoed back exactly, as
#: Google asks ("include the exact id of the function_call in the function_response").
LOCAL_CALL_ID_PREFIX = "local-call-"

#: How a system turn that is **not** at the head of the conversation reads to Gemini, which has no
#: system role inside ``contents``. A label rather than a silent re-role, so the model can tell the
#: harness's own notes (the per-wake brief, a step note) from a peer's words.
SYSTEM_LABEL = "[System]\n"

#: The longest value a credential-file *setting* may hold. A path is short; anything longer is
#: almost certainly the key itself pasted where its path belongs.
_PATH_LIMIT = 1024

#: The finish reasons that mean the model tried to call a function and produced nothing usable. A
#: retry very often succeeds, so they are the transient response class, never a silent turn.
_BROKEN_CALL_REASONS = frozenset({"MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL"})


class GoogleConfigError(ValueError):
    """The adapter's Vertex configuration is unusable — raised at construction, never mid-wake.

    `reason` is a stable slug a caller can file a fault under (the describer reports it as
    ``config:<reason>``): ``missing_credentials_file``, ``unreadable_credentials_file``,
    ``invalid_credentials_file``, ``missing_location``, ``missing_project``, or
    ``invalid_model_params``. The message never carries the file's contents.
    """

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def require_google_genai():
    """Import and return ``google.genai``, or raise a clear "no LLM, by design" error.

    The core depends on **no** vendor SDK — an ``AI_SDK=google-genai`` agent installs the extra
    (``pip install 'basecradle-harness[google-genai]'``). Without it the harness genuinely cannot
    reach a model, so this fails loud and actionable at provider construction.
    """
    try:
        from google import genai  # lazy: the core must import without the vendor SDK
        from google.genai import (  # noqa: F401 - imported so a broken install fails here
            errors,
            types,
        )
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised via monkeypatched import
        raise ProviderError(
            "The 'google-genai' SDK is not installed, so the harness has no way to reach a model "
            "(this is by design — the core depends on no vendor SDK). Install the SDK your "
            "agent's AI_SDK names: pip install 'basecradle-harness[google-genai]'."
        ) from exc
    return genai


def credentials_path(raw: str | os.PathLike[str] | None) -> Path | None:
    """`raw` as the path the credential is read from — a relative one under the config home.

    ``None`` for an unset or blank value. A relative path is the NOC's natural spelling for a secret
    it placed beside the agent's other configuration, so it resolves against the config home the
    rest of the agent's files already live in (`_install.config_home`), never against whatever
    directory the wake happened to start in.

    **A value that is the key itself is refused without being repeated** (`looks_like_a_path`). Many
    Google tools accept the JSON inline, so pasting it here is the obvious mistake — and every
    message below names the path, so an inline key would otherwise be written whole into a startup
    error, an ERROR line on every wake, and ``--resolved-config``.
    """
    text = str(raw).strip() if raw is not None else ""
    if not text:
        return None
    if not looks_like_a_path(text):
        raise GoogleConfigError(
            "inline_credentials",
            "The Vertex credential setting holds something that is not a file path (it looks like "
            "the key itself). Put the key in a file and set the variable to that file's path.",
        )
    path = Path(text).expanduser()
    if not path.is_absolute():
        from basecradle_harness._install import config_home

        path = config_home() / path
    return path


def looks_like_a_path(value: str) -> bool:
    """Whether a credential-file setting can be a path — never the key pasted in its place."""
    return bool(value) and len(value) <= _PATH_LIMIT and "{" not in value and "\n" not in value


def reported_credentials_file(raw: str | None) -> str | None:
    """A credential-file setting as ``--resolved-config`` may print it: the path, or a placeholder.

    The report is pasted into issues, so a value that is not a plausible path (the key itself) is
    never echoed; the placeholder says the setting is present and wrong, which is the fact a drift
    pass needs.
    """
    value = (raw or "").strip()
    if not value:
        return None
    return value if looks_like_a_path(value) else "[withheld: not a path]"


def load_credentials(path: Path) -> tuple[Any, str | None]:
    """Load a **service-account** credential from `path`: ``(credentials, project the file names)``.

    Every failure is a `GoogleConfigError` naming the path and never the contents: the file holds a
    private key, and an error message is the most-copied text on the box.
    """
    from google.oauth2 import service_account

    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise GoogleConfigError(
            "missing_credentials_file", f"The Vertex credential file {path} does not exist."
        ) from exc
    except OSError as exc:
        raise GoogleConfigError(
            "unreadable_credentials_file",
            f"The Vertex credential file {path} cannot be read ({type(exc).__name__}).",
        ) from exc
    try:
        info = json.loads(raw)
    except ValueError as exc:
        raise GoogleConfigError(
            "invalid_credentials_file",
            f"The Vertex credential file {path} is not valid JSON.",
        ) from exc
    if not isinstance(info, Mapping) or info.get("type") != "service_account":
        raise GoogleConfigError(
            "invalid_credentials_file",
            f"The Vertex credential file {path} is not a service-account key "
            '(its "type" must be "service_account").',
        )
    try:
        credentials = service_account.Credentials.from_service_account_info(
            info, scopes=[CLOUD_PLATFORM_SCOPE]
        )
    except (ValueError, KeyError, TypeError) as exc:
        raise GoogleConfigError(
            "invalid_credentials_file",
            f"The Vertex credential file {path} is not a usable service-account key "
            f"({type(exc).__name__}).",
        ) from None
    project = info.get("project_id")
    return credentials, project if isinstance(project, str) and project.strip() else None


class _Sealed:
    """A credential object held by this adapter, out of every representation of the adapter.

    `Secret` holds a string; a Google credentials object holds a private-key signer and, once
    refreshed, a bearer token in its ``__dict__``. This keeps the same promise for it (issue #599):
    no ``__dict__`` to walk, a redacted ``repr``, and no pickling.
    """

    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        self.value = value

    def __repr__(self) -> str:
        return "_Sealed('[REDACTED]')"

    __str__ = __repr__

    def __reduce__(self) -> Any:
        raise TypeError("A sealed credential cannot be serialized.")

    def __copy__(self) -> _Sealed:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> _Sealed:
        return self


#: The statuses Google's token endpoint answers when the fault is on its side or a capacity limit
#: — the ones ``google-auth`` would sleep and retry on. Here they end the refresh at once, as a
#: transport fault the engine's own retry decides on.
_TOKEN_RETRYABLE = frozenset({408, 429, 500, 502, 503, 504})


def _token_request(timeout: tuple[float, float]) -> Any:
    """A ``google-auth`` transport for **one** bounded token request, never retried inside it.

    Left to itself, the SDK refreshes the service-account token through ``google-auth``'s default
    transport: a 120-second timeout per attempt, and up to three attempts with backoff on a
    retryable status. That is a second retry policy under the engine's and an unbounded wait the
    fitted budget never sees (`CLAUDE.md` → Provider Capabilities: no adapter retries inside its
    SDK). So the adapter refreshes the token itself, through this: the fixed connect budget, a
    bounded read, a retryable status raised as a transport fault immediately, and any second
    request in one refresh refused.
    """
    from google.auth import exceptions as auth_errors
    from google.auth.transport import requests as auth_requests

    class _OneShot(auth_requests.Request):
        used = False

        def __call__(self, url, method="GET", body=None, headers=None, timeout=None, **kwargs):
            if self.used:
                raise auth_errors.TransportError(
                    "Google's token endpoint is not retried inside google-auth; the harness's own "
                    "retry decides."
                )
            self.used = True
            response = super().__call__(
                url, method=method, body=body, headers=headers, timeout=bounds, **kwargs
            )
            if response.status in _TOKEN_RETRYABLE:
                raise auth_errors.TransportError(
                    f"Google's token endpoint answered HTTP {response.status}."
                )
            return response

    bounds = timeout
    return _OneShot()


class _PhasedTimeout:
    """An ``httpx`` request hook that gives each Vertex call the harness's two budgets (issue #589).

    The SDK takes one timeout and spreads it over every phase of the request — connect and read
    alike — which is the flat wall #589 replaced. So the adapter hands the SDK its own ``httpx``
    client carrying this hook, and the hook stamps the phased budget (a fixed connect, the fitted
    generation for the read) onto the request just before it is sent. The SDK still receives the
    generation budget as its own timeout, which is what makes it tell Vertex the server-side
    deadline (``X-Server-Timeout``).
    """

    def __init__(self) -> None:
        self.budget: CallTimeout | None = None

    def __call__(self, request: httpx.Request) -> None:
        if self.budget is not None:
            request.extensions["timeout"] = httpx.Timeout(**self.budget.phases()).as_dict()


class GoogleProvider:
    """A `Provider` backed by Google's official ``google-genai`` SDK, in Vertex AI mode.

    Satisfies the `Provider` protocol — the engine cannot tell it from any other adapter — but every
    model call goes through ``google.genai``.

    Args:
        model: The Gemini model id (e.g. ``"gemini-3.8-flash"``).
        credentials_file: Path to a service-account JSON key; relative paths resolve against the
            config home (`credentials_path`). ``None`` reads the brain's whole configuration from
            the environment — ``AI_CREDENTIALS_FILE``, and ``AI_LOCATION`` / ``AI_PROJECT`` for
            whichever of the two below is not given. A caller passing its own key file gets none of
            the three: they describe the brain's credential, not this one.
        location: The Vertex location (``"us"``, ``"global"``, ``"us-central1"``, …); **required**
            — see the module docstring for why there is no default.
        project: The Google Cloud project; defaults to the one the key file names.
        base_url: An endpoint override (``AI_BASE_URL``); ``None`` lets the SDK choose the
            location's own endpoint (``aiplatform.us.rep.googleapis.com`` for ``us``).
        timeout: A **fixed** generation budget in seconds, overriding the fit — for a library caller
            that knows its calls. ``None`` (every deployment) fits each call (`_timeouts`).
        extra_body: Fields merged into the request body on every call (the SDK's own
            ``http_options.extra_body``) — the escape hatch for a field the typed config does not
            name yet, from the operator's ``model_params.json``.
        client: An already-built ``google.genai.Client`` (or compatible). The seam tests inject one;
            when given, no credential is loaded and nothing is built.
        builtin_tools: The server-side built-ins to send beside the function declarations on every
            turn, by builtin name (`BUILTINS`: ``code_execution``, ``url_context``). A name this
            adapter does not send is ignored — no plugin resolves one for this provider.
        default_params: ``GenerateContentConfig`` fields applied to every call, from the operator's
            ``model_params.json`` — ``temperature``, ``top_p``, ``top_k``, ``max_output_tokens``,
            ``thinking_config``, ``safety_settings``, ``seed``, ``stop_sequences``, ``labels``,
            ``service_tier``, and the rest of the typed set. Validated here, so a key the SDK does
            not name fails at startup rather than mid-wake.
    """

    #: Gemini on Vertex caches a repeated prefix **implicitly** (Gemini 2.5 and later) and reports
    #: the hit as ``cached_content_token_count``, which rides the line as ``cached_tokens=``. The
    #: engine marks nothing. Explicit context caching exists too, but it is a separate resource the
    #: operator would create, not a breakpoint the request carries.
    cache_mode = AUTOMATIC

    #: The ``AI_SDK`` this adapter is, and the one surface it speaks (issue #564).
    sdk = SDK
    surface = DEFAULT_SURFACE

    def __init__(
        self,
        model: str,
        *,
        credentials_file: str | os.PathLike[str] | None = None,
        location: str | None = None,
        project: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        extra_body: Mapping[str, Any] | None = None,
        client: Any | None = None,
        builtin_tools: Sequence[str] = (),
        **default_params: Any,
    ) -> None:
        self.model = model
        self.provider = PROVIDER
        #: The input-token count Vertex reported for the most recent call (issue #276).
        self.last_tokens_in: int | None = None
        #: Why Vertex stopped generating on that call, in its own words (issue #490).
        self.last_finish_reason: str | None = None
        #: The generation budget applied to the most recent call, in seconds (issue #589).
        self.last_timeout: float | None = None
        self._timeout_scale = 1.0
        self._fixed_timeout = timeout
        self._extra_body = dict(extra_body) if extra_body else None
        self._genai = require_google_genai()
        self._types = self._genai.types
        self._default_params, self._config = _validated_params(self._types, default_params)
        #: The thought signature each function call came back with, for the life of this adapter —
        #: one wake (see the module docstring for why it is not persisted). Keyed by the whole call
        #: (`_call_key`), never the vendor's id alone: an id is unique only within one response, so
        #: a model that numbers its calls per response would otherwise hand step 1's call step 2's
        #: signature.
        self._signatures: dict[tuple[str, str, str], bytes] = {}
        #: The same signatures by the **call object** the adapter returned — the first lookup, so a
        #: tool that mutates its own nested arguments in place cannot orphan its call's signature.
        #: The call is held beside its id so the id is never reused while the entry lives.
        self._signed_calls: dict[int, tuple[ToolCall, bytes]] = {}
        #: The models already warned about for having no rate in the table, so the warning is once
        #: per model per wake rather than once per call.
        self._unpriced: set[str] = set()
        #: The built-ins' wire entries, built once; sent ahead of the function declarations.
        self._builtins = [
            self._types.Tool(**{BUILTINS[name]: _builtin_config(self._types, name)})
            for name in dict.fromkeys(builtin_tools)
            if name in BUILTINS
        ]
        self._phased = _PhasedTimeout()
        self._http: httpx.Client | None = None
        #: The service-account credentials, sealed, so the adapter can refresh the token on its own
        #: bounded terms before each call (`_refresh_token`). ``None`` for an injected client.
        self._credentials: _Sealed | None = None
        if client is not None:
            self._client = client
            self.location = (location or os.environ.get(LOCATION_VAR) or "").strip().lower() or None
            self.project = project
            return
        # The `AI_*` variables are **one** configuration — the brain's — so they are read together or
        # not at all: only a caller that passes no key file gets the environment's key, location and
        # project. A caller that passes its own key file (the describer, on its own credential)
        # never has the brain's location or project slipped in beside it.
        from_env = credentials_file is None
        if from_env:
            credentials_file = os.environ.get(CREDENTIALS_FILE_VAR)
            location = location or os.environ.get(LOCATION_VAR)
            project = project or os.environ.get(PROJECT_VAR)
        # Lower-cased because the SDK compares it case-sensitively (``"US"`` would build a regional
        # host that does not exist, and fail every call as a dropped connection) while pricing does
        # not: the two must read the same location.
        location = (location or "").strip().lower()
        if not location:
            raise GoogleConfigError(
                "missing_location",
                f"No Vertex location: set {LOCATION_VAR} (e.g. 'us'). There is deliberately no "
                "default — the SDK's would be 'global', which lets Google process the call anywhere.",
            )
        path = credentials_path(credentials_file)
        if path is None:
            raise GoogleConfigError(
                "missing_credentials_file",
                f"No Vertex credential: set {CREDENTIALS_FILE_VAR} to the path of a "
                "service-account JSON key.",
            )
        credentials, file_project = load_credentials(path)
        project = (project or "").strip() or file_project
        if not project:
            raise GoogleConfigError(
                "missing_project",
                f"No Google Cloud project: set {PROJECT_VAR}, or use a credential file that names "
                "its project_id.",
            )
        self.location = location
        self.project = project
        #: The endpoint override this adapter was built with, or ``None`` for the location's own.
        self.base_url = base_url or None
        self._credentials = _Sealed(credentials)
        self._http = httpx.Client(follow_redirects=True, event_hooks={"request": [self._phased]})
        options: dict[str, Any] = {"httpx_client": self._http}
        if base_url:
            options["base_url"] = base_url
        self._client = self._genai.Client(
            vertexai=True,
            credentials=credentials,
            project=project,
            location=location,
            # No `retry_options`: the SDK's default is a single attempt, and the engine's bounded
            # retry is the one policy for every adapter (`_retry`). A second retry inside the SDK
            # would compose with it and re-send a timed-out request into the same wall.
            http_options=self._types.HttpOptions(**options),
        )
        _serializable_for_vertex(self._client, self._config)

    # --- capabilities ---------------------------------------------------------

    @property
    def tuning(self) -> dict[str, Any]:
        """What this adapter adds to every call — for an agent's brain, its ``model_params.json``."""
        tuned = copy.deepcopy(self._default_params)
        if self._extra_body:
            tuned["extra_body"] = copy.deepcopy(self._extra_body)
        return tuned

    def bind_timeout_scale(self, scale: float) -> None:
        """How much of its fitted generation budget this adapter's next calls get (issue #589)."""
        self._timeout_scale = max(1.0, float(scale))

    def supports_vision(self) -> bool | None:
        """``True`` for a Gemini model — every one takes image input — and unknown otherwise (#228).

        Google states the modalities per model page rather than in any API the SDK reads, and every
        Gemini ``generateContent`` model lists image, video and audio input. A model id outside the
        ``gemini-`` family is **unknown**, which the vision gate reads as "show the image" and the
        video gate as "send frames".
        """
        return True if _is_gemini(self.model) else None

    def supports_video(self) -> bool | None:
        """``True`` for a Gemini model, which watches video natively; unknown otherwise (#471)."""
        return True if _is_gemini(self.model) else None

    def context_limit(self) -> int | None:
        """``None``: Vertex states no context window this SDK can read (issue #276's capability).

        On Vertex, ``models.get`` returns the *publisher model* record, and the SDK's Vertex reader
        maps no token limit out of it (``input_token_limit`` is filled only on the Gemini Developer
        API) — so asking would spend a request on every wake to learn nothing. The honest answer is
        *unknown*, the same one the OpenAI adapter gives, and the context budget then takes its
        conservative floor. An operator who wants the model's real window sets
        ``HARNESS_MAX_CONTEXT_TOKENS`` from the model's page (1,048,576 for the Gemini 3 Flash
        family) — the documented override, never a table of this repo's own.
        """
        return None

    # --- the call -------------------------------------------------------------

    def chat(self, messages: Sequence[Message], tools: Sequence[ToolSpec] | None = None) -> Message:
        """Run one model turn through the SDK and return the assistant's reply."""
        budget = call_timeout(
            request_chars(messages, tools),
            output_tokens=output_cap(self._default_params),
            scale=self._timeout_scale,
            fixed=self._fixed_timeout,
        )
        self._phased.budget = budget
        self.last_timeout = budget.generation if self._http is not None else None
        system, contents = self._to_wire(messages)
        types = self._types
        update: dict[str, Any] = {
            # The SDK can call Python functions it is handed by itself; the harness hands it
            # declarations, never callables, and runs every tool through its own registry.
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
            "http_options": types.HttpOptions(
                timeout=int(budget.generation * 1000),
                extra_body=copy.deepcopy(self._extra_body) if self._extra_body else None,
            ),
        }
        if system:
            update["system_instruction"] = system
        declared = [self._declaration(spec) for spec in tools or ()]
        wire_tools = list(self._builtins)
        if declared:
            wire_tools.append(types.Tool(function_declarations=declared))
        if wire_tools:
            update["tools"] = wire_tools
        # The operator's tuning as the SDK's own typed config, so every nested field (a
        # `thinking_config`, a `safety_settings` list) is serialized the way the SDK serializes it.
        config = self._config.model_copy(update=update)
        started = time.monotonic()
        with _mapped_errors(self._genai):
            self._refresh_token(budget)
            response = self._client.models.generate_content(
                model=self.model, contents=contents, config=config
            )
        meta = getattr(response, "usage_metadata", None)
        usage = _wire_usage(meta)
        self.last_tokens_in = token_counts(usage).get("tokens_in")
        candidate = _first_candidate(response)
        reason = _finish_name(candidate)
        self.last_finish_reason = finish_reason({"finish_reason": reason})
        response_id = generation_id({"id": getattr(response, "response_id", None)})
        log_llm_call(
            provider=self.provider,
            model=self.model,
            seconds=time.monotonic() - started,
            usage=usage,
            # A direct-to-vendor SDK has no upstream to name, so the field carries where the call
            # ran: the residency fact an operator chose this provider for.
            endpoint=self.location,
            cost=self._cost(meta),
            finish_reason=self.last_finish_reason,
            generation_id=response_id,
        )
        with generation_of({"id": response_id}):
            return self._from_wire(response, candidate, reason)

    def search(self, query: str) -> str:
        """Answer `query` with Google Search grounding, in one call of its own (issue #656).

        The call's only tool is ``google_search`` — no function declarations, which Vertex will not
        combine with a search tool — and its only content is the query, so nothing of the agent's
        conversation leaves for it. Returns the grounded answer with a ``Sources:`` footer.

        Two lines, both ``kind=search.grounding``: the call's ``llm`` line as ``purpose=helper`` —
        it is a model call the brain made on the agent's behalf, never the brain's own turn — with
        its tokens priced from the table (grounding's own input tokens exempt where Google says so),
        and a ``media`` line for the grounding fee, which is not tokens and is the fleet's tool
        spend. A fee the table cannot state is logged without ``cost=`` and one WARNING. Raises the
        same typed `ProviderError`s as `chat`; the tool hands the model their text.
        """
        types = self._types
        budget = call_timeout(
            len(query), output_tokens=output_cap(self._default_params), fixed=self._fixed_timeout
        )
        self._phased.budget = budget
        config = self._config.model_copy(
            update={
                "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
                "http_options": types.HttpOptions(
                    timeout=int(budget.generation * 1000),
                    extra_body=copy.deepcopy(self._extra_body) if self._extra_body else None,
                ),
                "tools": [types.Tool(google_search=types.GoogleSearch())],
            }
        )
        contents = [types.Content(role="user", parts=[types.Part(text=query)])]
        started = time.monotonic()
        with _mapped_errors(self._genai):
            self._refresh_token(budget)
            response = self._client.models.generate_content(
                model=self.model, contents=contents, config=config
            )
        seconds = time.monotonic() - started
        meta = getattr(response, "usage_metadata", None)
        candidate = _first_candidate(response)
        log_llm_call(
            provider=self.provider,
            purpose=HELPER,
            kind=SEARCH_KIND,
            endpoint=self.location,
            model=self.model,
            seconds=seconds,
            usage=_wire_usage(meta),
            cost=self._cost(meta, grounded=True),
            outcome="ok",
            generation_id=generation_id({"id": getattr(response, "response_id", None)}),
        )
        grounding = getattr(candidate, "grounding_metadata", None)
        sources = _grounding_sources(grounding)
        queries = _grounding_queries(grounding)
        if sources:
            fee = grounding_cost(self.model, queries=queries, sourced=True)
            if fee is None:
                _log.warning(
                    "Google Search grounding on %r is not priced (no published scheme for the "
                    "model, or no grounding queries listed to count); its fee carries no cost=.",
                    self.model,
                )
            log_media_call(
                provider=self.provider,
                kind=SEARCH_KIND,
                model=self.model,
                seconds=None,
                count=grounding_units(self.model, queries),
                cost=fee,
            )
        parts = getattr(getattr(candidate, "content", None), "parts", None) or []
        text = "".join(
            part.text
            for part in parts
            if getattr(part, "text", None) and not getattr(part, "thought", None)
        )
        return (text or "(Google Search returned no answer.)") + format_citations(sources)

    def _refresh_token(self, budget: CallTimeout) -> None:
        """Refresh the access token now, bounded and single-shot, if it is not valid (`_token_request`).

        ``valid`` is the same test the SDK applies before its own refresh (a token, not within
        ``google-auth``'s refresh threshold of expiry), so a token this leaves valid is one the SDK
        will not refresh either, and its own unbounded refresh is not reached.
        """
        if self._credentials is None or self._credentials.value.valid:
            return
        self._credentials.value.refresh(_token_request((budget.connect, METADATA_TIMEOUT)))

    def _cost(self, meta: Any, *, grounded: bool = False) -> float | None:
        """This call's dollars at Google's published rates — computed, never vendor-stated (#655)."""
        if not known(self.model) and self.model not in self._unpriced:
            self._unpriced.add(self.model)
            _log.warning(
                "No published rate for Vertex model %r in the harness's table, so its calls carry "
                "no cost= on the llm line. Add the model's row to _google_rates.RATES from Google's "
                "pricing page.",
                self.model,
            )
        return call_cost(
            self.model,
            self.location or "",
            from_usage_metadata(meta),
            service_tier=self._default_params.get("service_tier"),
            grounded=grounded,
        )

    # --- harness -> Gemini ----------------------------------------------------

    def _to_wire(self, messages: Sequence[Message]) -> tuple[str | None, list[Any]]:
        """The conversation as ``(system_instruction, contents)``.

        See the module docstring for the three translations that matter: leading system turns only,
        tool results grouped and answered by name, and thought signatures on the current turn.
        """
        types = self._types
        leading: list[str] = []
        index = 0
        while index < len(messages) and messages[index].role == "system":
            if messages[index].content:
                leading.append(messages[index].content)
            index += 1
        contents: list[Any] = []
        names: dict[str, str] = {}  # call id → function name, from the latest assistant turn
        results: list[Any] = []
        #: The first call each model content carries, by its position in `contents` — what `_sign`
        #: looks a signature up by.
        first_calls: dict[int, ToolCall] = {}

        def flush_results() -> None:
            if results:
                contents.append(types.Content(role="user", parts=list(results)))
                results.clear()

        for message in messages[index:]:
            if message.role == "tool":
                call_id = message.tool_call_id or ""
                results.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            id=_wire_id(call_id),
                            name=names.get(call_id, "tool"),
                            response={"result": message.content or ""},
                        )
                    )
                )
                continue
            flush_results()
            if message.role == "assistant":
                names = {call.id: call.name for call in message.tool_calls}
                parts = []
                if message.content:
                    parts.append(types.Part(text=message.content))
                parts.extend(
                    types.Part(
                        function_call=types.FunctionCall(
                            id=_wire_id(call.id), name=call.name, args=call.arguments
                        )
                    )
                    for call in message.tool_calls
                )
                if parts:
                    if message.tool_calls:
                        first_calls[len(contents)] = message.tool_calls[0]
                    contents.append(types.Content(role="model", parts=parts))
                continue
            parts = []
            if message.role == "system":
                if message.content:
                    parts.append(types.Part(text=SYSTEM_LABEL + message.content))
            else:
                if message.content:
                    parts.append(types.Part(text=message.content))
                parts.extend(self._media_part(image) for image in message.images)
                parts.extend(self._media_part(video) for video in message.videos)
            if parts:
                contents.append(types.Content(role="user", parts=parts))
        flush_results()
        self._sign(contents, first_calls)
        return ("\n\n".join(leading) or None), contents

    def _sign(self, contents: list[Any], first_calls: Mapping[int, ToolCall]) -> None:
        """Put each thought signature this adapter holds back on the call it came with.

        Two of Google's rules decide what goes where. The first ``functionCall`` part of every step
        in the **current turn** must carry a signature or the request is a 400, and "a turn begins
        with the most recent user message that is not a functionResponse". And beyond that minimum,
        signatures "from previous responses" should be sent back so the model keeps its reasoning.

        So every signature still held goes back on its call, wherever the call sits — which matters
        here more than the rule's minimum suggests, because the engine appends a step note after a
        step's results, and to Google that note is a new user turn: signing only the current turn
        would send back none of the signatures the model actually made. A current-turn call with no
        signature held (its turn was begun by an earlier wake, now resumed) carries `SKIP_SIGNATURE`;
        an older one goes without, which is what Google's validator expects of history.
        """
        current = 0
        for position in range(len(contents) - 1, -1, -1):
            content = contents[position]
            if content.role == "user" and not any(p.function_response for p in content.parts):
                current = position
                break
        for position, call in first_calls.items():
            held = self._signed_calls.get(id(call))
            signature = (
                held[1] if held and held[0] is call else self._signatures.get(_call_key(call))
            )
            if signature is None and position < current:
                continue
            first = next(p for p in contents[position].parts if p.function_call is not None)
            first.thought_signature = signature or SKIP_SIGNATURE

    def _declaration(self, spec: ToolSpec) -> Any:
        """A harness tool as a Gemini ``FunctionDeclaration``, its JSON Schema passed as written."""
        fields: dict[str, Any] = {"name": spec.name, "description": spec.description}
        if spec.parameters:
            fields["parameters_json_schema"] = spec.parameters
        return self._types.FunctionDeclaration(**fields)

    def _media_part(self, media: ImageContent | VideoContent) -> Any:
        """An image or a clip as a Gemini part — inline bytes for a data URL, a reference otherwise.

        The harness inlines everything it shows as a ``data:`` URL, so the reference branch exists
        for a library caller's ``https``/``gs`` URL, whose media type must then be knowable from
        the URL itself: a part Google would have to guess the type of is refused here, loudly.
        """
        types = self._types
        url = media.url
        if url.startswith("data:"):
            header, _, payload = url.partition(",")
            mime = header[len("data:") :].split(";", 1)[0] or None
            if mime is None or ";base64" not in header:
                raise ProviderError(f"Unsupported inline media URL for Vertex: {header[:64]!r}.")
            return types.Part.from_bytes(data=base64.b64decode(payload), mime_type=mime)
        mime = getattr(media, "content_type", None) or mimetypes.guess_type(url)[0]
        if not mime:
            raise ProviderError(
                f"Cannot send {url[:80]!r} to Vertex: its media type is not knowable from the URL."
            )
        return types.Part.from_uri(file_uri=url, mime_type=mime)

    # --- Gemini -> harness ----------------------------------------------------

    def _from_wire(self, response: Any, candidate: Any, reason: str | None) -> Message:
        """The first candidate as a harness assistant `Message` (text + function calls).

        Thought-summary parts (``thought=True``, present only when an operator asks for
        ``include_thoughts``) are the model's reasoning, not its reply, and are left out.
        """
        parts = getattr(getattr(candidate, "content", None), "parts", None) or []
        texts: list[str] = []
        calls: list[ToolCall] = []
        for part in parts:
            if getattr(part, "thought", None):
                continue
            call = getattr(part, "function_call", None)
            if call is not None:
                call_id = getattr(call, "id", None) or f"{LOCAL_CALL_ID_PREFIX}{uuid.uuid4().hex}"
                signature = getattr(part, "thought_signature", None)
                made = ToolCall(id=call_id, name=call.name or "", arguments=dict(call.args or {}))
                if signature:
                    self._signatures[_call_key(made)] = signature
                    self._signed_calls[id(made)] = (made, signature)
                calls.append(made)
                continue
            text = getattr(part, "text", None)
            if text:
                texts.append(text)
        if not calls and not texts:
            if reason in _BROKEN_CALL_REASONS:
                raise ProviderResponseError(
                    f"Vertex returned no usable reply (finish_reason={reason}); retrying."
                )
            blocked = _block_reason(response) or reason
            if blocked and blocked != "STOP":
                _log.warning(
                    "Vertex returned an empty reply for %s (reason=%s); the turn ends silent.",
                    self.model,
                    blocked,
                )
        return Message.assistant(content="".join(texts) or None, tool_calls=calls)

    # --- lifecycle --------------------------------------------------------------

    def close(self) -> None:
        """Close the ``httpx`` client this adapter built (an injected client is the caller's)."""
        if self._http is not None:
            self._http.close()

    def __enter__(self) -> GoogleProvider:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# --- helpers --------------------------------------------------------------------


def _validated_params(types: Any, params: Mapping[str, Any]) -> tuple[dict[str, Any], Any]:
    """`params` as written, and as the SDK's own ``GenerateContentConfig`` — built at startup.

    The typed config forbids unknown fields, so a misspelt ``model_params.json`` key fails here
    rather than inside a wake's model call. The typed form is what each call is built from: handed a
    plain mapping, the SDK forwards a nested field's keys as the operator spelled them, while its
    typed models are serialized the way the API documents. The mapping is kept for `tuning`, so the
    brief reports what the operator wrote.
    """
    try:
        typed = types.GenerateContentConfig(**params)
    except Exception as exc:
        raise GoogleConfigError(
            "invalid_model_params",
            f"model_params.json is not accepted by the google-genai SDK's GenerateContentConfig: "
            f"{exc}",
        ) from exc
    return dict(params), typed


def _builtin_config(types: Any, name: str) -> Any:
    """The (empty) configuration object a built-in's ``types.Tool`` field takes."""
    return {"code_execution": types.ToolCodeExecution, "url_context": types.UrlContext}[name]()


def _grounding_sources(grounding: Any) -> list[dict[str, str]]:
    """The web sources a grounded answer cites, as ``{"url", "title"}`` for `format_citations`."""
    sources = []
    for chunk in getattr(grounding, "grounding_chunks", None) or ():
        web = getattr(chunk, "web", None)
        uri = getattr(web, "uri", None)
        if uri:
            sources.append({"url": uri, "title": getattr(web, "title", None) or ""})
    return sources


def _grounding_queries(grounding: Any) -> int | None:
    """How many grounding queries Google lists for the call — ``None`` when it lists none."""
    queries = [q for q in getattr(grounding, "web_search_queries", None) or () if q]
    return len(queries) or None


def _call_key(call: ToolCall) -> tuple[str, str, str]:
    """A function call's identity across a wake: its id, name and arguments together."""
    return (call.id, call.name, json.dumps(call.arguments, sort_keys=True, default=str))


def _serializable_for_vertex(client: Any, config: Any) -> None:
    """Run the SDK's own Vertex serializer over the operator's tuning once, at startup.

    The typed config accepts fields the SDK then refuses to send to Vertex ("only supported in
    Gemini Developer API mode" — ``enable_enhanced_civic_answers``, ``tool_config``'s
    ``include_server_side_tool_invocations``, a replicated voice's consent fields, and whatever a
    later SDK adds). Asking the serializer itself catches every one of them, where a hand-kept list
    would go stale on the next SDK bump. It is a private SDK function, so its absence (a renamed
    internal) skips the check, and the per-call mapping (`_mapped_errors`) still turns the same
    refusal into a provider error rather than a raw ``ValueError``.
    """
    from google.genai import models as sdk_models

    serialize = getattr(sdk_models, "_GenerateContentConfig_to_vertex", None)
    api_client = getattr(client, "_api_client", None)
    if serialize is None or api_client is None:
        return
    try:
        serialize(api_client, config, {})
    except ValueError as exc:
        raise GoogleConfigError(
            "invalid_model_params",
            f"model_params.json holds a setting the google-genai SDK will not send to Vertex AI: "
            f"{exc}",
        ) from exc


def _is_gemini(model: str) -> bool:
    return model.strip().rsplit("/", 1)[-1].lower().startswith("gemini-")


def _wire_id(call_id: str) -> str | None:
    """The id to send back for a call: the model's own, or nothing for one the adapter made up."""
    return None if not call_id or call_id.startswith(LOCAL_CALL_ID_PREFIX) else call_id


def _first_candidate(response: Any) -> Any:
    candidates = getattr(response, "candidates", None) or []
    return candidates[0] if candidates else None


def _finish_name(candidate: Any) -> str | None:
    value = getattr(candidate, "finish_reason", None)
    if value is None:
        return None
    return str(getattr(value, "value", value)).upper() or None


def _block_reason(response: Any) -> str | None:
    feedback = getattr(response, "prompt_feedback", None)
    value = getattr(feedback, "block_reason", None)
    if value is None:
        return None
    return str(getattr(value, "value", value)).upper() or None


def _wire_usage(meta: Any) -> dict[str, Any] | None:
    """Vertex's ``usage_metadata`` in the harness's usage vocabulary (the chat-wire spelling).

    One translation, here at the vendor boundary, so the shared readers need no Gemini branch:
    ``tokens_in`` is the prompt (its cached part included), ``tokens_out`` is the reply **plus** the
    thinking, because thinking is billed as output exactly as a reasoning model's is on every other
    wire, and ``tokens_reasoning`` is the thinking alone.
    """
    if meta is None:
        return None
    prompt = getattr(meta, "prompt_token_count", None)
    reply = getattr(meta, "candidates_token_count", None)
    thoughts = getattr(meta, "thoughts_token_count", None)
    if prompt is None and reply is None and thoughts is None:
        return None
    usage: dict[str, Any] = {}
    if isinstance(prompt, int):
        usage["prompt_tokens"] = prompt
    if isinstance(reply, int) or isinstance(thoughts, int):
        usage["completion_tokens"] = (reply or 0) + (thoughts or 0)
    total = getattr(meta, "total_token_count", None)
    if isinstance(total, int):
        usage["total_tokens"] = total
    cached = getattr(meta, "cached_content_token_count", None)
    if isinstance(cached, int):
        usage["prompt_tokens_details"] = {"cached_tokens": cached}
    if isinstance(thoughts, int):
        usage["completion_tokens_details"] = {"reasoning_tokens": thoughts}
    return usage


class _mapped_errors:
    """Map ``google-genai``, ``google-auth`` and ``httpx`` failures onto the harness's typed errors.

    Classified by the nature of the fault, never by vendor (`CLAUDE.md` → Provider Capabilities):
    a context overflow compacts and retries, an oversized payload is reported once, an account
    without billing is the billing class, a quota refusal is a rate limit, a 5xx is the provider's
    own side, and a transport failure or timeout is transient. A generic 400 propagates as a plain
    `ProviderAPIError`, so a fixable config defect never loses the peer's message.
    """

    def __init__(self, genai: Any) -> None:
        self._genai = genai

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is None:
            return False
        errors = self._genai.errors
        if isinstance(exc, errors.APIError):
            raise _from_api_error(exc) from exc
        if isinstance(exc, errors.UnknownApiResponseError):
            raise ProviderResponseError(f"Vertex returned an unparseable response: {exc}") from exc
        from google.auth import exceptions as auth_errors

        if isinstance(exc, auth_errors.RefreshError):
            if getattr(exc, "retryable", False):
                raise ProviderConnectionError(
                    f"Could not refresh the Vertex access token: {exc}"
                ) from exc
            raise ProviderAuthError(
                f"Google refused the Vertex service-account credential: {exc}", status_code=401
            ) from exc
        if isinstance(exc, auth_errors.TransportError):
            raise ProviderConnectionError(
                f"Could not reach Google's token endpoint: {exc}"
            ) from exc
        if isinstance(exc, httpx.TimeoutException):
            raise ProviderTimeoutError(f"Vertex did not answer in time: {exc}") from exc
        if isinstance(exc, httpx.RequestError):
            raise ProviderConnectionError(f"Could not reach Vertex: {exc}") from exc
        if isinstance(exc, ValueError):
            # The SDK refusing to build the request (a setting it will not send to Vertex) — a
            # configuration fault, so a plain provider error that propagates and leaves the peer's
            # message re-drivable, never a raw ValueError out of the adapter.
            raise ProviderError(f"The google-genai SDK refused the request: {exc}") from exc
        return False


def _from_api_error(exc: Any) -> ProviderError:
    """A ``google.genai.errors.APIError`` as the right typed `ProviderError`."""
    status = getattr(exc, "code", None) or 0
    message = getattr(exc, "message", None) or str(exc)
    details = getattr(exc, "details", None)
    body = json.dumps(details) if isinstance(details, (Mapping, list)) else str(details or "")
    text = f"{message} {body}"
    retry_after = _retry_after(getattr(exc, "response", None))
    if status in (400, 413) and is_context_overflow(text):
        return ProviderContextLengthError(message, status_code=status, body=body)
    if status == 413 or (status == 400 and is_too_large(text)):
        return ProviderPayloadTooLargeError(message, status_code=status, body=body)
    if status in (401, 403):
        if _billing_disabled(details) or is_out_of_funds(text):
            return ProviderBillingError(message, status_code=status, body=body)
        return ProviderAuthError(
            f"Vertex refused the call (HTTP {status}): {message}", status_code=status, body=body
        )
    if status == 429:
        return ProviderRateLimitError(
            f"Vertex rate-limited the request (HTTP 429): {message}",
            status_code=status,
            body=body,
            retry_after=retry_after,
        )
    if status == 504 or getattr(exc, "status", None) == "DEADLINE_EXCEEDED":
        # Vertex's own deadline — the `X-Server-Timeout` the SDK sends with the fitted budget — ran
        # out before the answer was ready. It is the server-side twin of a read timeout, so it is
        # typed as one: retried once with twice the budget (issue #589), never re-sent into the same
        # wall as a plain 5xx would be. The server's clock starts on arrival and the client's read
        # timer only after the upload, so on a large request this is the one that fires first.
        return ProviderTimeoutError(f"Vertex's deadline ran out (HTTP {status}): {message}")
    if status >= 500:
        return ProviderServerError(
            f"Vertex failed on its own side (HTTP {status}).",
            status_code=status,
            body=body,
            retry_after=retry_after,
        )
    return ProviderAPIError(message, status_code=status, body=body)


def _billing_disabled(details: Any) -> bool:
    """Whether Google's structured error says the project's billing is off (``BILLING_DISABLED``)."""
    error = details.get("error") if isinstance(details, Mapping) else None
    for entry in (error or {}).get("details", ()) if isinstance(error, Mapping) else ():
        if isinstance(entry, Mapping) and entry.get("reason") == "BILLING_DISABLED":
            return True
    return False


def _retry_after(response: Any) -> float | None:
    headers = getattr(response, "headers", None)
    raw = headers.get("retry-after") if headers is not None else None
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None
