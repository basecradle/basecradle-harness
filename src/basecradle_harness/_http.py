"""HTTP error mapping for the xAI media tools (`_grok`), which call ``api.x.ai`` with ``httpx``.

No provider adapter uses this module: every adapter reaches its vendor through the vendor's
SDK and maps that SDK's errors itself (`_openai._from_status_error` and its siblings). Every
`ProviderError` a media tool raises is relayed to the model the same way, so this mapping names
only the auth and rate-limit statuses and leaves every other status a plain `ProviderAPIError`.
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
