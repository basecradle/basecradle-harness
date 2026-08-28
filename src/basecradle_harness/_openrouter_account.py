"""Read the agent's own OpenRouter credit runway — cost self-awareness (issue #425).

The OpenRouter mirror of `_xai_account.py`: same shape, same soft-failure contract, same
locked-profile-safe plumbing — a plain read-only function `Tool` (no platform client, no policy
capability, no shell) that makes one authenticated HTTPS GET and returns a figure. An agent
holding an OpenRouter account can see its remaining credit and reason about its runway: throttle,
prioritize cheap experiments, or ask a human to top up *before* it runs dry as a hard API failure.

It talks to `GET /api/v1/credits` with a dedicated **Management key**
(`OPENROUTER_MANAGEMENT_KEY`) — an account-administration credential distinct from the inference
`AI_API_KEY`, minted at ``openrouter.ai/settings/management-keys``. An ordinary inference key is
rejected there (verified live, 2026-08-28: HTTP 401), which is exactly why the credential is its
own variable and never `AI_API_KEY`.

**The arithmetic is a subtraction, and it is the whole of it.** The endpoint returns two
*lifetime cumulative* totals — ``data.total_credits`` (purchased to date) and ``data.total_usage``
(used to date) — as plain positive JSON numbers, and the remaining credit is their difference.
None of the xAI Management API's traps exist here: no cents strings, no inverted sign convention,
no team-UUID discovery, and **no posted-ledger vs. invoice-preview dichotomy** — one endpoint,
one figure. Do not go looking for a second surface or invent a degraded tier; there is nothing to
fall back *to*, so an unusable response is simply an unavailable one.

Two consequences of "lifetime cumulative" that are easy to get wrong and are load-bearing here:

- **The context line says "to date", never "this billing cycle".** These totals do not reset at
  cycle close, so cycle wording would be false — and a runway figure the model misreads as a
  *cycle* budget is the same class of defect as xAI's stale posted ledger (issue #384): a number
  that means something other than what it appears to.
- **The difference can legitimately go negative.** Usage can overrun purchased credit, so the
  overdraft callout is kept from the xAI tool rather than assumed impossible.

The figures are **quantized to cents at the parse boundary**, so the subtraction the tool *shows*
is the subtraction it *did*: the two terms and the headline are mutually consistent by
construction, and a model that re-does the arithmetic off the context line lands on the headline.
(``total_usage`` carries sub-cent precision — ``268.464928179`` in a live reading — which is
noise at runway scale and would otherwise let a rounded display contradict an unrounded headline,
or render a four-tenths-of-a-cent overrun as an alarming ``-$0.00`` overdraft.)

Everything fails **soft**: a missing key, the wrong kind of key, an unreachable endpoint, or an
unexpected response all return a clear ``"unavailable — <reason>"`` string rather than raising,
so a billing check never derails a wake. And it is careful with what it exposes: it never logs or
returns the key, and it never surfaces a response *body* — OpenRouter's 4xx envelope carries an
``error.message`` and a ``user_id`` that are account detail, so only computed figures leave here.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

from basecradle_harness._tools import NO_PARAMETERS, Tool

#: The OpenRouter API root. `/credits` is an account-administration surface reached with a
#: Management key, not the inference credential — the same endpoint host, a different key.
#: Overridable only for a proxy or a test double, never to reach another vendor.
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

#: Per-request HTTP timeout (seconds). A billing read is a quick GET, and a balance check must
#: never block a wake for long, so this is short — a timeout is a ceiling, not a fixed wait.
DEFAULT_TIMEOUT = 15.0

#: How long a fetched figure is reused before re-reading (seconds). Credit moves slowly, and a
#: model reasoning about its runway may check it more than once in a turn; a short cache keeps
#: those from each hitting the billing endpoint. ``0`` disables caching entirely.
DEFAULT_CACHE_TTL = 30.0


class _BalanceUnavailable(Exception):
    """An internal signal carrying a model-readable reason a figure couldn't be read.

    Raised by the request/parse helpers and turned into the tool's ``"unavailable — <reason>"``
    return string by `run`. Its message is always safe to show the model — it names *what* went
    wrong (a status code, a shape problem), never the key or the raw payload.
    """


@dataclass(frozen=True)
class _Credits:
    """The account's two lifetime totals, in dollars and cents, and the runway they imply.

    `remaining_usd` is the answer the tool exists to give, and it is a **subtraction** — so it is
    *derived* rather than stored, and the headline figure can never disagree with the two terms
    shown behind it. Both terms are already quantized to cents (see `_parse_credits`), which is
    what makes that consistency hold in the rendering as well as in the arithmetic.
    """

    purchased_usd: float
    used_usd: float

    @property
    def remaining_usd(self) -> float:
        """Credits purchased to date, net of usage to date — what is left to spend."""
        return self.purchased_usd - self.used_usd


class OpenRouterAccountBalanceTool(Tool):
    """`openrouter_account_balance` — report the credit remaining on the agent's own OpenRouter
    account.

    A plain read-only `Tool` (no platform client, no policy capability) that calls
    ``GET /api/v1/credits`` with a dedicated `OPENROUTER_MANAGEMENT_KEY`, so an agent holding an
    OpenRouter account can see its remaining credit and reason about its runway. The figure is
    ``total_credits − total_usage``: purchased to date, less used to date.

    Unlike its xAI sibling its plugin declares **no `Vendor` requirement**, deliberately (issue
    #425). The credential is dedicated and provider-independent, and the ordering case is an agent
    brained by *another* provider that nonetheless holds an OpenRouter account — a
    ``Vendor("openrouter")`` gate would self-exclude exactly that agent. It stays `opt_in` like
    every powerful tool (issue #168), because it reaches an account/billing surface.

    It degrades to a clear ``"OpenRouter account balance unavailable — <reason>"`` string in every
    failure mode (no key, an inference key instead of a Management key, endpoint unreachable,
    unexpected response) rather than raising. It never logs or returns the key or a response body.

    Args:
        management_key: The Management key. ``None`` (the default) reads
            `OPENROUTER_MANAGEMENT_KEY` from the environment at call time.
        base_url: The API root (for a proxy or a test double).
        timeout: Per-request timeout in seconds.
        cache_ttl: Seconds to reuse a fetched figure before re-reading; ``0`` disables it.
        clock: The monotonic clock the cache measures against (injectable for tests).
    """

    name = "openrouter_account_balance"
    description = (
        "Check the credit remaining on your own OpenRouter account right now, in US dollars — "
        "the credits purchased on the account to date, less what has been used to date. Use it "
        "to reason about your runway: throttle or prioritize cheap work when credit is low, or "
        "ask a human to top up before you run dry mid-task. Takes no arguments and can see only "
        "your own account, nothing else."
    )
    parameters = NO_PARAMETERS

    def __init__(
        self,
        *,
        management_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        cache_ttl: float = DEFAULT_CACHE_TTL,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._management_key = management_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._cache_ttl = cache_ttl
        self._clock = clock
        # (monotonic expiry, the key it was read with, rendered text). The key is part of the
        # entry because it is resolved from the environment at *call* time: without it, a figure
        # read for one account — with that account's `as of` stamp — is served as another's for
        # the rest of the TTL after the credential is repointed.
        self._cached: tuple[float, str, str] | None = None

    def run(self) -> str:
        """Report the credit remaining, or a clear reason it is unavailable."""
        key = self._management_key or os.environ.get("OPENROUTER_MANAGEMENT_KEY")
        if not key:
            return (
                "OpenRouter account balance unavailable — OPENROUTER_MANAGEMENT_KEY is not "
                "configured. Set it to an OpenRouter Management key "
                "(openrouter.ai/settings/management-keys → Create New Key); an ordinary "
                "inference API key is not accepted on this endpoint."
            )

        cached = self._cached_balance(key)
        if cached is not None:
            return cached

        try:
            # The client is built *inside* the guard, not outside it: httpx encodes a header
            # value as ASCII at construction time, so a key carrying a smart quote, a
            # non-breaking space or an accented character — the ordinary consequence of pasting
            # a credential — raises `UnicodeEncodeError` before any request exists to fail. A
            # misconfigured credential is the case this tool most owes a soft answer to.
            with httpx.Client(
                headers={"Authorization": f"Bearer {key}"}, timeout=self._timeout
            ) as client:
                credits = self._credits(client)
        except UnicodeEncodeError:
            return (
                "OpenRouter account balance unavailable — OPENROUTER_MANAGEMENT_KEY contains a "
                "non-ASCII character, so it cannot be sent as a header; check for a smart quote "
                "or a non-breaking space picked up when it was pasted."
            )
        except _BalanceUnavailable as exc:
            return f"OpenRouter account balance unavailable — {exc}"

        text = _render(credits, self._now_utc())
        if self._cache_ttl > 0:
            self._cached = (self._clock() + self._cache_ttl, key, text)
        return text

    # --- the one HTTP call ----------------------------------------------------

    def _credits(self, client: httpx.Client) -> _Credits:
        """The account's lifetime purchased/used totals — the only surface this tool reads."""
        data = self._get(client, "/credits", "read the credits remaining")
        return _parse_credits(data)

    def _get(self, client: httpx.Client, path: str, action: str) -> dict[str, Any]:
        """GET an API path and decode it, mapping every failure to a soft reason.

        Never surfaces the response *body*: OpenRouter's error envelope carries an
        ``error.message`` and a ``user_id``, and the contract is that only computed figures ever
        leave this tool.
        """
        try:
            response = client.get(f"{self._base_url}{path}")
        except httpx.RequestError as exc:
            raise _BalanceUnavailable(f"couldn't reach the OpenRouter API ({exc}).") from None

        status = response.status_code
        if status in (401, 403):
            raise _BalanceUnavailable(
                f"OpenRouter rejected the key (HTTP {status}); this endpoint needs a Management "
                "key, not an inference API key. Check OPENROUTER_MANAGEMENT_KEY."
            )
        if status >= 400:
            raise _BalanceUnavailable(f"OpenRouter returned HTTP {status} trying to {action}.")
        try:
            data = response.json()
        except (ValueError, RecursionError):
            # `json` raises `RecursionError` — not a `ValueError` — on a deeply nested body, so
            # a hostile or broken intermediary could otherwise raise straight out of `run()`.
            raise _BalanceUnavailable(
                f"OpenRouter returned an unreadable response trying to {action}."
            ) from None
        if not isinstance(data, dict):
            raise _BalanceUnavailable(
                f"OpenRouter returned an unexpected response trying to {action}."
            )
        return data

    # --- caching --------------------------------------------------------------

    def _cached_balance(self, key: str) -> str | None:
        """The cached text if it is still fresh **and** was read with this key, else ``None``."""
        if self._cached is None:
            return None
        expiry, cached_key, text = self._cached
        if cached_key == key and self._clock() < expiry:
            return text
        return None

    def _now_utc(self) -> datetime:
        """Wall-clock now, in UTC — the ``as of`` stamp on a freshly-read figure."""
        return datetime.now(timezone.utc)


def _parse_credits(data: dict[str, Any]) -> _Credits:
    """The two lifetime totals out of ``GET /credits``.

    Shape: ``{"data": {"total_credits": 500.0, "total_usage": 123.456789}}`` — plain positive
    JSON numbers of USD, verified live 2026-08-28. Note the live reading returned
    ``total_credits`` as a bare integer (``375``), so an integer is as valid here as a float;
    ``bool`` is not, despite being an `int` subclass in Python.

    **Both fields are required**, because the answer is their difference: purchased-to-date alone
    is not a runway (an account that has bought $500 and spent $500 has none), and a response
    carrying only one of them cannot answer the question at all. The tempting "tolerant" parse
    that reports ``total_credits`` when the usage is missing is xAI's issue #388 defect, one
    vendor over.

    Both are quantized to **cents** so the rendered subtraction is self-consistent — see the
    module docstring.
    """
    inner = data.get("data")
    if not isinstance(inner, dict):
        raise _BalanceUnavailable("OpenRouter's credits response carried no data object.")
    purchased = _usd(inner, "total_credits")
    used = _usd(inner, "total_usage")
    if purchased is None or used is None:
        missing = "total_credits" if purchased is None else "total_usage"
        raise _BalanceUnavailable(
            f"OpenRouter's credits response carried no usable {missing} figure; the remaining "
            "credit is total_credits minus total_usage, so both are needed."
        )
    return _Credits(purchased_usd=purchased, used_usd=used)


def _usd(holder: dict[str, Any], field: str) -> float | None:
    """The dollars at ``holder[field]``, rounded to cents, or ``None`` if absent or unusable.

    Three rejections, and none of them is defensive padding — each is a way a JSON body reaches
    Python as something that is *not* a dollar figure:

    - **``bool``** passes ``isinstance(x, int)`` and would quietly become ``$1.00`` of credit.
    - **A non-finite float.** Python's `json` accepts the non-standard ``NaN`` / ``Infinity``
      literals by default, and neither survives contact with the renderer: ``nan < 0`` is `False`,
      so an overdraft check silently says healthy, and both format as a *figure* — ``$nan USD``,
      ``$inf USD``. The contract is that only computed figures leave this tool, and those are not
      figures; an unreadable response is the honest answer.
    - **An integer too large to be a float.** A JSON integer literal has no width limit, so a
      400-digit one parses fine and then ``float()`` raises `OverflowError` — *out of* ``run()``,
      which is the one thing this tool must never do.

    A numeric *string* is rejected too — the vendor sends numbers, and inventing a tolerance for a
    shape it does not send would hide a real contract drift behind a figure nobody checked.
    """
    figure = holder.get(field)
    if isinstance(figure, bool) or not isinstance(figure, (int, float)):
        return None
    try:
        usd = float(figure)
    except OverflowError:
        return None
    return round(usd, 2) if math.isfinite(usd) else None


def _dollars(usd: float) -> str:
    """A signed USD figure — ``$42.50`` / ``-$5.00``.

    **Both arms format the magnitude.** The positive arm formatting ``usd`` directly is the
    obvious shape and it is wrong, because `round` manufactures ``-0.0`` out of anything in
    ``[-0.005, 0)`` and ``-0.0 < 0`` is `False` — so a value that *is* zero to the cent falls
    through to the positive arm and renders as ``$-0.00``: the sign on the wrong side of the
    dollar, outside the uniform headline shape every reader, regex and the live probe depend on,
    and precisely the alarming near-zero "overdraft" the cent quantization exists to eliminate.
    Formatting ``abs(usd)`` on both sides makes the rendered sign agree with the rendered
    magnitude by construction.
    """
    return f"-${abs(usd):,.2f}" if usd < 0 else f"${abs(usd):,.2f}"


def _stamp(as_of: datetime) -> str:
    """The ``as of`` timestamp, ISO-8601 to the second with a ``Z`` suffix."""
    return as_of.isoformat(timespec="seconds").replace("+00:00", "Z")


def _render(credits: _Credits, as_of: datetime) -> str:
    """The headline figure, with the subtraction that produced it shown underneath.

    The headline shape is **uniform** — ``<label>: <$figure> USD (as of <stamp>).`` — so the
    figure is always in the same place and a reader (or a regex, or the model) never has to parse
    around a parenthetical that appears only sometimes; an overdraft is called out in the body
    instead, where it leads so it is not buried.

    The subtraction is **shown**, not merely performed: an agent that meets ``total_credits``
    elsewhere — OpenRouter's own dashboard, another client — can see which number that is and why
    it is the larger one. The wording is "to date" on both terms because these are lifetime
    cumulative totals that never reset; calling either one a billing cycle's would be false.
    """
    context = ["The account is overdrawn."] if credits.remaining_usd < 0 else []
    context.append(
        f"Live figure — {_dollars(credits.purchased_usd)} of credits purchased to date less the "
        f"{_dollars(credits.used_usd)} used to date."
    )
    headline = (
        f"OpenRouter credits remaining: {_dollars(credits.remaining_usd)} USD "
        f"(as of {_stamp(as_of)})."
    )
    return f"{headline}\n" + " ".join(context)
