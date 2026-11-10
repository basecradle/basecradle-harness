"""Read the agent's own refuseless credit — cost self-awareness on a prepaid brain (issue #848).

The refuseless sibling of `_openrouter_account.py` and `_xai_account.py`: the same soft-failure
contract and the same locked-profile-safe plumbing — a plain read-only function `Tool` (no platform
client, no policy capability, no shell) that makes one authenticated HTTPS GET and returns figures.
An agent brained on refuseless pays for every call from prepaid credit, so it can see what it has
and reason about its runway: throttle, prioritize cheap work, or ask a human to top up before a call
fails for want of credit.

It reads ``GET <base_url>/credits`` with the agent's **own inference key**, ``AI_API_KEY``, at the
host its calls go to (``AI_BASE_URL``, else the vendor's default), and it is why it is gated on
``Vendor("refuseless")``: the brain's credential is a refuseless key **only when the brain is
refuseless**. The xAI sibling reads a *dedicated* account credential and so is gated only on its
vendor's account; the OpenRouter sibling also reads OpenRouter keys that serve helpers (the
reranker, the describer), which is its whole case for having no vendor gate, and it checks
``AI_PROVIDER`` itself before it reads the brain's key (issue #853). This tool has only the
brain's. Offered beside any other brain it would send that vendor's key to refuseless's host,
which is a credential sent to a stranger, not a balance read. An agent with a separate refuseless
account beside another brain would need a dedicated variable for that key, which nothing asks for.

**The figures are the vendor's, never arithmetic of ours.** The call is free (no credit charged, no
hold placed) and answers::

    {"object": "credits", "balance": 69.989469, "reserved": 0, "available": 69.989469,
     "key_limit": null, "key_spent": 0.001405, "key_spent_since": null, "unit": "usd",
     "updated_at": 1794353525}

``balance`` is the account's total credit, including what in-flight requests hold; ``reserved`` is
that hold (the vendor pre-authorizes each request and returns what it does not use when the request
settles); ``available`` is what a new request can reserve, balance less reserved and capped by
``key_limit``; ``key_limit`` is what this key's budget (or its workspace's limit) still allows, or
``null`` when it has none; ``key_spent`` is what this key has been billed, in total or since
``key_spent_since`` when that is not ``null`` (issue #868). ``available`` is the headline, because it is the runway: a request is
refused when it cannot reserve, whatever the balance says. Each figure is shown **to the vendor's
own precision** — US dollars to six decimal places, which is as far as the vendor states them —
because none of them is derived here, so there is no subtraction to keep self-consistent, and
rounding to cents would only make a stated figure disagree with the vendor's own dashboard. ``updated_at`` (the moment the vendor read the balance) is
the ``as of`` stamp. The balance fields are required, as the vendor documents them; a response
missing one, in another unit, or carrying a value that is not a finite number is *unavailable*,
never a guess. **The key's spend is not one of them** (issue #868): the vendor added ``key_spent``
and ``key_spent_since`` later, so a response without a usable pair still answers with the balance,
and only the sentence about the key's spend is left out. A pair is usable when ``key_spent`` is a
finite, non-negative number and ``key_spent_since`` is ``null`` or a non-negative integer number of
seconds; a figure whose span cannot be read is not reported, since the span is what makes it a
figure.

Everything fails **soft**: no key, a rejected key, the rate limit (60 balance reads a minute per
key), the vendor's ``gateway_down`` (its billing service unreachable), any other non-2xx, an
unreachable host, an unusable ``AI_BASE_URL`` or an unexpected shape each return a clear
``"unavailable — <reason>"`` and record the failure on the ``tool`` line (`tool_failed`), a status
named by `docs/telemetry.md` → Fault Slugs. It never logs, returns or keeps the key in plain view
(the figure cache is keyed on a digest of it), and never surfaces a response *body*: the error
envelope's ``message`` is the vendor's prose about the account, so only the error ``code`` is read,
and only to recognize ``gateway_down``.

**The gate is enforced twice.** The plugin is gated on ``Vendor("refuseless")``, and the tool itself
refuses to read ``AI_API_KEY`` or ``AI_BASE_URL`` from the environment unless ``AI_PROVIDER`` is
``refuseless``, so a library caller that builds it beside another brain cannot send that brain's key
to refuseless's host, or a refuseless key to that brain's. A caller that hands it both explicitly
reads nothing from the environment.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx

from basecradle_harness._basecradle import (
    _PROVIDER_BASE_URLS,
    base_url_override,
    normalize_provider,
)
from basecradle_harness._observability import tool_failed
from basecradle_harness._secret import reveal, secret
from basecradle_harness._tools import NO_PARAMETERS, Tool

#: The vendor this tool reads an account of — the `AI_PROVIDER` its plugin is gated on.
VENDOR = "refuseless"

#: Per-request HTTP timeout (seconds). A balance read is a quick GET and must never hold a wake.
DEFAULT_TIMEOUT = 15.0

#: How long a fetched figure is reused before re-reading (seconds). The vendor allows 60 balance
#: reads a minute per key, so a model checking its runway more than once in a turn reuses the figure
#: rather than spending that allowance. ``0`` disables caching.
DEFAULT_CACHE_TTL = 30.0

#: The one ``unit`` every figure must be stated in. A figure in an unstated unit is not a dollar.
UNIT = "usd"

#: The vendor's error code for its billing service being unreachable (documented on ``/credits``).
GATEWAY_DOWN = "gateway_down"


class _BalanceUnavailable(Exception):
    """A model-readable reason the figures could not be read, and the `tool` line's slug for it.

    Its message is always safe to show the model: it names *what* went wrong (a status, a shape),
    never the key or the vendor's own error prose.
    """

    def __init__(self, message: str, *, slug: str = "provider_error") -> None:
        super().__init__(message)
        #: The `tool` line's `reason` for this failure (issue #822).
        self.slug = slug


@dataclass(frozen=True)
class _Credits:
    """The account's figures exactly as the vendor stated them, in US dollars."""

    balance: float
    reserved: float
    available: float
    #: ``None`` when the key has no budget of its own.
    key_limit: float | None
    #: When the vendor read the balance.
    updated_at: datetime
    #: What this key has been billed, or ``None`` when the vendor stated no usable figure for it.
    key_spend: _KeySpend | None = None


@dataclass(frozen=True)
class _KeySpend:
    """What the calling key has been billed, in US dollars, as the vendor stated it (issue #868)."""

    spent: float
    #: When the vendor's counter started; ``None`` when it states ``null``, so ``spent`` is the
    #: key's whole total.
    since: datetime | None


class RefuselessAccountBalanceTool(Tool):
    """`refuseless_account_balance` — report the credit on the agent's own refuseless account.

    A plain read-only `Tool` (no platform client, no policy capability) that calls
    ``GET <base_url>/credits`` with the agent's ``AI_API_KEY``. Its plugin is gated on
    ``Vendor("refuseless")``, because that key is the brain's and is a refuseless key only when the
    brain is (module docstring). It is a benign default since issue #862 (@origin, 2026-11-10: every
    agent reads the balance at its main LLM vendor when the read exposes nothing else sensitive), so
    that gate alone decides where it appears.

    Args:
        api_key: The refuseless key. ``None`` (the default) reads ``AI_API_KEY`` at call time.
        base_url: The API root. ``None`` (the default) reads ``AI_BASE_URL`` at call time and falls
            back to the vendor's own host, which is exactly where the brain's calls go.
        timeout: Per-request timeout in seconds.

    Either of `api_key` and `base_url` left ``None`` is read from the environment, which the tool
    does only when ``AI_PROVIDER`` is ``refuseless`` (module docstring).
        cache_ttl: Seconds to reuse a fetched figure before re-reading; ``0`` disables it.
        clock: The monotonic clock the cache measures against (injectable for tests).
    """

    name = "refuseless_account_balance"
    description = (
        "Check the credit on your own refuseless account right now, in US dollars: what a new "
        "request can still spend, the account balance, the part of it held by requests in flight, "
        "and this key's own spending limit if it has one. Use it to reason about your runway: "
        "throttle or prioritize cheap work when credit is low, or ask a human to top up before you "
        "run dry mid-task. It also says what this key has been billed. Takes no arguments, "
        "costs nothing, and can see only your own account."
    )
    parameters = NO_PARAMETERS

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        cache_ttl: float = DEFAULT_CACHE_TTL,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._api_key = secret(api_key)  # never a plain attribute (issue #599)
        self._base_url = base_url
        self._timeout = timeout
        self._cache_ttl = cache_ttl
        self._clock = clock
        # (monotonic expiry, digest of the key, root, rendered text). The key and the root are part
        # of the entry because both are resolved at *call* time: without them, one account's figures
        # would be served as another's for the rest of the TTL after either is repointed. A digest,
        # never the key: a plain attribute holding it is emitted by `vars()` and every crash
        # reporter that expands locals (issue #599).
        self._cached: tuple[float, bytes, str, str] | None = None

    def run(self) -> str:
        """Report the credit, or a clear reason it is unavailable."""
        if reveal(self._api_key) is None or self._base_url is None:
            provider = _active_provider()
            if provider != VENDOR:
                return tool_failed(
                    f"refuseless account balance unavailable — AI_PROVIDER is {provider!r}, not "
                    "'refuseless', so AI_API_KEY is not a refuseless key and AI_BASE_URL is not "
                    "refuseless's host. This tool reads the account of a refuseless brain only.",
                    reason="config:provider",
                )
        key = reveal(self._api_key) or (os.environ.get("AI_API_KEY") or "").strip()
        if not key:
            return tool_failed(
                "refuseless account balance unavailable — AI_API_KEY is not configured, so there "
                "is no refuseless key to read the account with.",
                reason="config:missing_api_key",
            )
        root = self._root()
        digest = hashlib.sha256(key.encode("utf-8", "surrogatepass")).digest()

        cached = self._cached_balance(digest, root)
        if cached is not None:
            return cached

        try:
            # Built inside the guard: httpx encodes a header as ASCII at construction, so a pasted
            # key carrying a smart quote raises before any request exists to fail.
            with httpx.Client(
                headers={"Authorization": f"Bearer {key}"}, timeout=self._timeout
            ) as client:
                credits = _parse_credits(self._get(client, f"{root}/credits"))
        except httpx.InvalidURL:
            return tool_failed(
                "refuseless account balance unavailable — AI_BASE_URL is not a usable URL, so "
                "there is no host to read the account at.",
                reason="config:base_url",
            )
        except UnicodeEncodeError:
            return tool_failed(
                "refuseless account balance unavailable — AI_API_KEY contains a non-ASCII "
                "character, so it cannot be sent as a header; check for a smart quote or a "
                "non-breaking space picked up when it was pasted.",
                reason="config:auth",
            )
        except _BalanceUnavailable as exc:
            return tool_failed(f"refuseless account balance unavailable — {exc}", reason=exc.slug)

        text = _render(credits)
        if self._cache_ttl > 0:
            self._cached = (self._clock() + self._cache_ttl, digest, root, text)
        return text

    def _root(self) -> str:
        """The API root this call reads: the one the brain's own calls go to."""
        root = self._base_url or base_url_override() or _PROVIDER_BASE_URLS[VENDOR]
        return root.rstrip("/")

    def _get(self, client: httpx.Client, url: str) -> dict[str, Any]:
        """GET ``/credits`` and decode it, mapping every failure to a soft reason."""
        try:
            response = client.get(url)
        except httpx.TimeoutException:
            raise _BalanceUnavailable(
                "the refuseless API did not answer in time.", slug="timeout"
            ) from None
        except httpx.RequestError as exc:
            raise _BalanceUnavailable(
                f"couldn't reach the refuseless API ({type(exc).__name__}).", slug="transport"
            ) from None

        status = response.status_code
        if status in (401, 403):
            raise _BalanceUnavailable(
                f"refuseless rejected the key (HTTP {status}). Check AI_API_KEY.",
                slug="config:auth",
            )
        if status == 429:
            raise _BalanceUnavailable(
                "refuseless is rate-limiting balance reads (HTTP 429; it allows 60 a minute per "
                "key). Try again in a minute.",
                slug="rate_limited",
            )
        if not 200 <= status < 300:
            slug = _STATUS_SLUGS.get(status) or ("server_error" if status >= 500 else "api_error")
            if _error_code(response) == GATEWAY_DOWN:
                raise _BalanceUnavailable(
                    f"refuseless's billing service is unreachable (HTTP {status}, gateway_down). "
                    "Try again shortly.",
                    slug=slug,
                )
            raise _BalanceUnavailable(
                f"refuseless returned HTTP {status} reading the credit balance.", slug=slug
            )
        try:
            data = response.json()
        except (ValueError, RecursionError):
            # `json` raises `RecursionError`, not a `ValueError`, on a deeply nested body.
            raise _BalanceUnavailable(
                "refuseless returned an unreadable response reading the credit balance.",
                slug="invalid_response",
            ) from None
        if not isinstance(data, dict):
            raise _BalanceUnavailable(
                "refuseless returned an unexpected response reading the credit balance.",
                slug="invalid_response",
            )
        return data

    def _cached_balance(self, digest: bytes, root: str) -> str | None:
        """The cached text if it is still fresh **and** was read with this key at this root."""
        if self._cached is None:
            return None
        expiry, cached_digest, cached_root, text = self._cached
        same_key = hmac.compare_digest(cached_digest, digest)
        if same_key and cached_root == root and self._clock() < expiry:
            return text
        return None


#: The statuses `docs/telemetry.md` → Fault Slugs names by the status alone, beyond the two this
#: tool words for itself (``401``/``403`` and ``429``). Any other ``5xx`` is ``server_error`` and
#: any other status (a ``3xx`` httpx did not follow included) is ``api_error``, as that table rules.
_STATUS_SLUGS = {
    402: "config:billing",
    404: "config:model_not_found",
    413: "payload_too_large",
    504: "timeout",
}


def _active_provider() -> str:
    """``AI_PROVIDER`` as the wake reads it, or the raw value when it is not even a slug."""
    raw = os.environ.get("AI_PROVIDER")
    try:
        return normalize_provider(raw)
    except ValueError:
        return (raw or "").strip()


def _error_code(response: httpx.Response) -> str | None:
    """The error envelope's ``code``, read only to recognize a documented one; never its message."""
    try:
        body = response.json()
    except (ValueError, RecursionError):
        return None
    error = body.get("error") if isinstance(body, dict) else None
    code = error.get("code") if isinstance(error, dict) else None
    return code if isinstance(code, str) else None


def _parse_credits(data: dict[str, Any]) -> _Credits:
    """The figures out of ``GET /credits``, every balance field required and the key's spend not.

    Unavailable for another ``object`` or ``unit`` (``usd``, as documented: a figure in an unstated
    unit is not a dollar), a dollar figure that is not a finite number, a ``key_limit`` that is
    neither a number nor ``null`` (an absent one included: the vendor states ``null`` for "no
    budget", so a missing field is not that statement), and an ``updated_at`` that is not a
    non-negative integer number of seconds.
    """
    if data.get("object") != "credits":
        raise _BalanceUnavailable(
            "refuseless's credits response was not a credits object.", slug="invalid_response"
        )
    if data.get("unit") != UNIT:
        raise _BalanceUnavailable(
            "refuseless's credits response did not state its figures in US dollars.",
            slug="invalid_response",
        )
    figures = {field: _usd(data.get(field)) for field in ("balance", "reserved", "available")}
    missing = [field for field, value in figures.items() if value is None]
    if "key_limit" not in data:
        missing.append("key_limit")
    key_limit = data.get("key_limit")
    limit = None if key_limit is None else _usd(key_limit)
    if key_limit is not None and limit is None:
        missing.append("key_limit")
    stamp = data.get("updated_at")
    if isinstance(stamp, bool) or not isinstance(stamp, int) or stamp < 0:
        missing.append("updated_at")
    if missing:
        raise _BalanceUnavailable(
            f"refuseless's credits response carried no usable {', '.join(missing)}.",
            slug="invalid_response",
        )
    try:
        updated_at = datetime.fromtimestamp(stamp, UTC)
    except (OverflowError, OSError, ValueError):
        raise _BalanceUnavailable(
            "refuseless's credits response carried no usable updated_at.",
            slug="invalid_response",
        ) from None
    return _Credits(
        balance=figures["balance"],  # type: ignore[arg-type]
        reserved=figures["reserved"],  # type: ignore[arg-type]
        available=figures["available"],  # type: ignore[arg-type]
        key_limit=limit,
        updated_at=updated_at,
        key_spend=_key_spend(data),
    )


def _key_spend(data: dict[str, Any]) -> _KeySpend | None:
    """The key's own spend out of ``GET /credits``, or ``None`` when the vendor stated none usably.

    Never required (module docstring): an absent or ``null`` ``key_spent``, one that is not a
    finite, non-negative number, and a ``key_spent_since`` that is neither ``null`` (absent
    included) nor a non-negative integer number of seconds each leave the spend out and the balance
    standing. A negative figure is refused because a key's billed total cannot be one.
    """
    spent = _usd(data.get("key_spent"))
    if spent is None or spent < 0:
        return None
    raw = data.get("key_spent_since")
    if raw is None:
        return _KeySpend(spent=spent, since=None)
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        return None
    try:
        since = datetime.fromtimestamp(raw, UTC)
    except (OverflowError, OSError, ValueError):
        return None
    return _KeySpend(spent=spent, since=since)


def _usd(figure: Any) -> float | None:
    """A JSON number as dollars, or ``None`` for anything that is not a finite one.

    A ``bool`` is an `int` to Python and would become a dollar; Python's `json` accepts ``NaN`` and
    ``Infinity``, which format as figures and compare as nothing; an integer too large for a float
    raises `OverflowError`, out of ``run()``. A numeric string is not what the vendor sends.
    """
    if isinstance(figure, bool) or not isinstance(figure, (int, float)):
        return None
    try:
        usd = float(figure)
    except OverflowError:
        return None
    return usd if math.isfinite(usd) else None


def _dollars(usd: float) -> str:
    """A signed USD figure as the vendor stated it: cents always, up to six decimals.

    ``69.999906`` → ``$69.999906``; ``70`` → ``$70.00``; ``0.1`` → ``$0.10``. Formatted from the
    magnitude on both arms, so a negative zero never renders as ``$-0.00``.
    """
    digits = f"{abs(usd):,.6f}".rstrip("0")
    whole, _, fraction = digits.partition(".")
    text = f"{whole}.{fraction.ljust(2, '0')}"
    return f"-${text}" if usd < 0 and text.strip("0.,") else f"${text}"


#: Below this, a figure renders as ``$0.00`` or less (`_dollars` shows six decimals), so the callout
#: fires on exactly the figures that read as nothing.
_HALF_MICRODOLLAR = 0.0000005


def _stamp(as_of: datetime) -> str:
    """The ``as of`` timestamp, ISO-8601 to the second with a ``Z`` suffix."""
    return as_of.isoformat(timespec="seconds").replace("+00:00", "Z")


def _render(credits: _Credits) -> str:
    """The headline figure (``available``), with every figure behind it as the vendor stated it.

    The headline shape is uniform — ``<label>: <$figure> USD (as of <stamp>).`` — so the figure is
    always in the same place; a credit that cannot pay for a request is called out first in the body.
    """
    context = []
    if credits.available < _HALF_MICRODOLLAR:  # nothing a request could reserve, as shown
        context.append("No new request can be paid for until the account is topped up.")
    context.append(
        f"Balance {_dollars(credits.balance)}, of which {_dollars(credits.reserved)} is held by "
        "requests in flight (an unused hold returns to the balance when its request settles)."
    )
    if credits.key_limit is None:
        context.append("This key has no spending limit of its own.")
    else:
        context.append(f"This key's spending limit allows {_dollars(credits.key_limit)} more.")
    spend = credits.key_spend
    if spend is not None:
        span = "in total" if spend.since is None else f"since {_stamp(spend.since)}"
        context.append(f"This key has been billed {_dollars(spend.spent)} {span}.")
    headline = (
        f"refuseless credit available: {_dollars(credits.available)} USD "
        f"(as of {_stamp(credits.updated_at)})."
    )
    return f"{headline}\n" + " ".join(context)
