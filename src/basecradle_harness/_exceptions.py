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


class ProviderAuthError(ProviderAPIError):
    """The provider rejected the API key (HTTP 401/403)."""


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
