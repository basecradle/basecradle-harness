"""The Harness error hierarchy.

`HarnessError` is the root every framework error descends from — catch it to
catch everything Harness raises. Provider failures form one branch; future
subsystems (tools, the engine) add their own under the same root.
"""

from __future__ import annotations


class HarnessError(Exception):
    """Root of every error Harness raises."""


class PolicyError(HarnessError):
    """A tool was rejected by the active policy (e.g. it needs a forbidden capability)."""


class PlatformError(HarnessError):
    """A platform-aware tool could not act on the platform.

    Most often: the tool was invoked before a hosting agent
    (`TimelineAgent`/`WakeAgent`) bound its live `PlatformContext`, so it has no
    SDK client or current timeline to act through.
    """


class EngineError(HarnessError):
    """The agent loop could not produce a final reply (e.g. it hit the step limit)."""


class ProviderError(HarnessError):
    """A model provider call failed."""


class ProviderResponseError(ProviderError):
    """The provider *answered*, but with a response the SDK could not parse.

    A truncated body, malformed JSON, or a schema mismatch (the "EOF while parsing a
    value" class observed on GLM-5.2/OpenRouter, issue #259). The response arrived — so
    this is not a `ProviderConnectionError` — but could not be turned into a turn. It is
    the one provider failure the engine **retries**: it is transient (the same call
    re-issued usually succeeds), and deliberately distinct from a *permanent*
    `ProviderError` (a bad `model_params.json` key, a missing SDK) that retrying would
    only repeat. Adapters map their SDK's response-parse/validation failure to this so the
    retry is provider-agnostic — classified by the *nature of the fault*, never the vendor.
    """


class ProviderConnectionError(ProviderError):
    """The provider could not be reached (DNS, TCP, TLS, timeout)."""


class ProviderAPIError(ProviderError):
    """The provider returned an error status.

    `status_code` is the HTTP status; `body` is the raw response text, kept for
    debugging since provider error schemas are not standardized.
    """

    def __init__(self, message: str, *, status_code: int, body: str = "") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class ProviderServerError(ProviderAPIError):
    """The provider failed on *its own side* (HTTP 5xx) — transient, and therefore retried.

    The second member of the engine's retryable class, alongside `ProviderResponseError`. A 5xx is
    the provider saying *"my fault, not yours"*: the request was well-formed, nothing about it will
    be improved by changing it, and re-issuing the identical call is exactly the right response.
    That is the opposite of a 4xx, where repeating the request only repeats the rejection.

    **Why this is a decision and not a detail.** Before it existed, whether a 5xx was retried was an
    accident of which SDK an agent happened to run: the ``openai`` SDK retries 5xx internally
    (``max_retries``), while the native ``openrouter`` adapter disables its SDK's retry outright (its
    Speakeasy default backs off for up to an hour, which would hang a wake) — so the *same* transient
    fault was silently retried on one provider and fatal on another, decided by nobody. Mapping it to
    a shared class moves the policy up into the engine, where it is uniform, bounded
    (`HARNESS_RESPONSE_RETRIES` + backoff), and stated.

    **What it costs to get wrong.** A wake marks each item *seen* **before** it calls the model, so a
    hard-failed wake does not merely fail — it **drops the peer's message permanently**: no reply,
    and no later wake to retry it. Against that, a bounded retry costs cents. (The retry narrows that
    window; it does not close it — see the `seen`-ordering defect tracked separately.)
    """


class ProviderAuthError(ProviderAPIError):
    """The provider rejected the API key (HTTP 401/403)."""


class ProviderContextLengthError(ProviderAPIError):
    """The request exceeded the model's context window — the wall (issue #276).

    Deterministic, not transient: the same transcript re-sent produces the same 400 forever, so
    retrying it unchanged only repeats it. It is separated from every other status error precisely
    *because* the harness can do something about it — `Session.send` catches it, compacts the
    transcript hard, and re-runs the turn once, so an agent that has already grown past its ceiling
    **self-heals on its next wake** instead of needing a human to edit its session file by hand.

    Adapters map their own over-length status error to this (see `is_context_overflow`), so the
    recovery is provider-agnostic — classified by the nature of the fault, never by the vendor.
    """


class ProviderRateLimitError(ProviderAPIError):
    """The provider rate-limited the request (HTTP 429).

    `retry_after` is the seconds hinted by the `Retry-After` header, if present.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        body: str = "",
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message, status_code=status_code, body=body)
        self.retry_after = retry_after
