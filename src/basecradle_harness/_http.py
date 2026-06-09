"""Shared HTTP plumbing for the OpenAI provider adapters.

Both provider adapters — the OpenAI-compatible Chat Completions one and the
OpenAI Responses one — talk to the same family of endpoints and get the same
error envelopes back. The mapping from an HTTP error status onto Harness's typed
provider exceptions is therefore one thing, owned here, not copied per adapter.
"""

from __future__ import annotations

import httpx

from basecradle_harness._exceptions import (
    ProviderAPIError,
    ProviderAuthError,
    ProviderRateLimitError,
)


def raise_for_status(response: httpx.Response) -> None:
    """Map an error response onto the typed provider exceptions."""
    status = response.status_code
    body = response.text
    if status in (401, 403):
        raise ProviderAuthError(
            f"Provider rejected the API key (HTTP {status}).", status_code=status, body=body
        )
    if status == 429:
        raise ProviderRateLimitError(
            "Provider rate-limited the request (HTTP 429).",
            status_code=status,
            body=body,
            retry_after=retry_after(response),
        )
    raise ProviderAPIError(f"Provider returned HTTP {status}.", status_code=status, body=body)


def retry_after(response: httpx.Response) -> float | None:
    """The seconds hinted by a `Retry-After` header, if present and numeric."""
    raw = response.headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None
