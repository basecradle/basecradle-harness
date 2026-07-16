"""Read the agent's own xAI prepaid credit balance — cost self-awareness (issue #179).

An xAI persona whose charter treats capital as a first-class concern can see its remaining
runway and reason about it — throttle, prioritize cheap experiments, or ask a human to top up
*before* it runs dry as a hard API failure. This is the tool that gives it that sense.

It talks to the xAI **Management API** (`management-api.x.ai`), a billing/account surface
distinct from the inference endpoint (`api.x.ai`), with its own dedicated credential — a
read-only **Management Key** (`XAI_MANAGEMENT_KEY`), never the agent's inference `AI_API_KEY`.
So it is a plain, read-only function `Tool` (no platform client, no policy capability, no shell)
that makes one authenticated HTTPS GET and returns a figure — the same locked-profile-safe shape
as `web_fetch`.

Two things the live API taught us that the naive sketch got wrong (both verified against a real
account, issue #179):

- **The balance lives at `total.val` as a string of USD *cents*** — not `total.amount`/
  `total.value`. And the **sign is inverted**: credit *added* (a PURCHASE) is stored negative,
  credit *spent* positive, so the available balance in dollars is the *negated* cents / 100 (the
  docs' own example: `val "-1000"` = `$10.00`). Getting this wrong reports a healthy positive
  balance as a negative one.
- **The team path segment is a UUID, not the literal `"default"`** — that 400s (`Invalid uuid`).
  The key knows its own team, so the tool *discovers* it from the management-key validation
  endpoint; `XAI_TEAM_ID` is an optional override, not a required (and misleading) default.

Everything fails **soft**: a missing key, the wrong scope, an unreachable endpoint, or an
unexpected response all return a clear "balance unavailable — <reason>" string rather than
raising, so a billing check never derails a wake. And it is careful with what it exposes: it
never logs or returns the key, and it never returns the raw billing payload (whose `changes`
ledger carries purchase/invoice history) — only the one computed balance figure the agent needs.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import httpx

from basecradle_harness._tools import NO_PARAMETERS, Tool

#: The xAI Management API root — a billing/account surface, distinct from the inference endpoint
#: (`api.x.ai`). Overridable only for a proxy or a test double, never to reach another vendor.
DEFAULT_BASE_URL = "https://management-api.x.ai"

#: Per-request HTTP timeout (seconds). A billing read is a quick GET, and a balance check must
#: never block a wake for long, so this is short — a timeout is a ceiling, not a fixed wait.
DEFAULT_TIMEOUT = 15.0

#: How long a fetched balance is reused before re-reading (seconds). A prepaid balance moves
#: slowly, and a model reasoning about its runway may check it more than once in a turn; a short
#: cache keeps those from each hitting the billing endpoint. ``0`` disables caching entirely.
DEFAULT_CACHE_TTL = 30.0


class _BalanceUnavailable(Exception):
    """An internal signal carrying a model-readable reason the balance couldn't be read.

    Raised by the request/parse helpers and turned into the tool's ``"unavailable — <reason>"``
    return string by `run`. Its message is always safe to show the model — it names *what* went
    wrong (a status code, a shape problem), never the key or the raw payload.
    """


class XaiAccountBalanceTool(Tool):
    """`xai_account_balance` — report this agent's own xAI prepaid credit balance.

    A plain read-only `Tool` (no platform client, no policy capability) that calls the xAI
    Management API with a dedicated `XAI_MANAGEMENT_KEY` (read-only billing scope), so an xAI
    agent can see its remaining credit and reason about its runway. Its plugin gates it to the
    xAI provider (`Vendor("xai")`) and marks it opt-in like every powerful tool.

    It degrades to a clear ``"xAI account balance unavailable — <reason>"`` string in every
    failure mode (no key, wrong scope, endpoint unreachable, unexpected response) rather than
    raising. It never logs or returns the key or the raw billing payload — only the computed
    balance figure.

    Args:
        management_key: The Management Key. ``None`` (the default) reads `XAI_MANAGEMENT_KEY`
            from the environment at call time.
        team_id: The team UUID. ``None`` reads `XAI_TEAM_ID`, and if that is unset too the team
            is discovered from the key itself (and cached on the instance).
        base_url: The Management API root (for a proxy or a test double).
        timeout: Per-request timeout in seconds.
        cache_ttl: Seconds to reuse a fetched balance before re-reading; ``0`` disables it.
        clock: The monotonic clock the cache measures against (injectable for tests).
    """

    name = "xai_account_balance"
    description = (
        "Check the real-time prepaid credit balance of your own xAI account, in US dollars. Use "
        "it to reason about your runway — throttle or prioritize cheap work when credit is low, "
        "or ask a human to top up before you run dry mid-task. Takes no arguments and can see "
        "only your own account's balance, nothing else. (When an X Developer account is linked, "
        "X API credit purchases grant free xAI credits, so this figure can reflect X-side spend "
        "rewards too.)"
    )
    parameters = NO_PARAMETERS

    def __init__(
        self,
        *,
        management_key: str | None = None,
        team_id: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        cache_ttl: float = DEFAULT_CACHE_TTL,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._management_key = management_key
        self._team_id = team_id
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._cache_ttl = cache_ttl
        self._clock = clock
        self._resolved_team: str | None = None
        self._cached: tuple[float, str] | None = None  # (monotonic expiry, balance text)

    def run(self) -> str:
        """Fetch and format the prepaid balance, or a clear reason it is unavailable."""
        key = self._management_key or os.environ.get("XAI_MANAGEMENT_KEY")
        if not key:
            return (
                "xAI account balance unavailable — XAI_MANAGEMENT_KEY is not configured. Set it "
                "to a read-only xAI Management Key (console.x.ai → Settings → Management Keys)."
            )

        cached = self._cached_balance()
        if cached is not None:
            return cached

        try:
            with httpx.Client(
                headers={"Authorization": f"Bearer {key}"}, timeout=self._timeout
            ) as client:
                team = self._team(client)
                balance_usd = self._balance(client, team)
        except _BalanceUnavailable as exc:
            return f"xAI account balance unavailable — {exc}"
        except httpx.RequestError as exc:
            return (
                f"xAI account balance unavailable — couldn't reach the xAI Management API ({exc})."
            )

        text = _render(balance_usd, self._now_utc())
        if self._cache_ttl > 0:
            self._cached = (self._clock() + self._cache_ttl, text)
        return text

    # --- the two HTTP calls ---------------------------------------------------

    def _team(self, client: httpx.Client) -> str:
        """The team UUID: an explicit override, else discovered from the key (then cached).

        The balance endpoint's path segment is a team *UUID* — the literal ``"default"`` 400s
        (``Invalid uuid``). The management-key validation endpoint reports the team the key acts
        for, so a correctly-scoped key needs no `XAI_TEAM_ID` at all; the override just skips the
        discovery call.
        """
        override = self._team_id or os.environ.get("XAI_TEAM_ID")
        if override:
            return override
        if self._resolved_team is not None:
            return self._resolved_team

        response = client.get(f"{self._base_url}/auth/management-keys/validation")
        data = self._json(response, "validate the management key")
        team = data.get("teamId")
        if not isinstance(team, str) or not team:
            raise _BalanceUnavailable(
                "the xAI Management API did not report a team for this key. Set XAI_TEAM_ID to "
                "your team UUID to bypass discovery."
            )
        self._resolved_team = team
        return team

    def _balance(self, client: httpx.Client, team: str) -> float:
        """Read the prepaid balance for `team` and return it in USD dollars."""
        response = client.get(f"{self._base_url}/v1/billing/teams/{team}/prepaid/balance")
        data = self._json(response, "read the prepaid balance")
        return _parse_balance_usd(data)

    def _json(self, response: httpx.Response, action: str) -> dict[str, Any]:
        """Decode a Management API response to JSON, mapping failures to soft reasons.

        Never surfaces the response *body*: a 4xx billing body can echo account detail, and the
        contract is that only the computed balance figure ever leaves this tool.
        """
        status = response.status_code
        if status in (401, 403):
            raise _BalanceUnavailable(
                f"the xAI Management API rejected the key (HTTP {status}); it needs read-only "
                "billing scope (BillingRead). Check XAI_MANAGEMENT_KEY."
            )
        if status >= 400:
            raise _BalanceUnavailable(
                f"the xAI Management API returned HTTP {status} trying to {action}."
            )
        try:
            data = response.json()
        except ValueError:
            raise _BalanceUnavailable(
                f"the xAI Management API returned an unreadable response trying to {action}."
            ) from None
        if not isinstance(data, dict):
            raise _BalanceUnavailable(
                f"the xAI Management API returned an unexpected response trying to {action}."
            )
        return data

    # --- caching --------------------------------------------------------------

    def _cached_balance(self) -> str | None:
        """The cached balance text if still fresh, else ``None``."""
        if self._cached is None:
            return None
        expiry, text = self._cached
        if self._clock() < expiry:
            return text
        return None

    def _now_utc(self) -> datetime:
        """Wall-clock now, in UTC — the ``as of`` stamp on a freshly-read balance."""
        return datetime.now(timezone.utc)


def _parse_balance_usd(data: dict[str, Any]) -> float:
    """The available prepaid balance in USD dollars, from the Management API's cents ledger.

    The endpoint returns ``{"total": {"val": "<cents>"}, "changes": [...]}`` where ``val`` is a
    **string** of USD cents whose **sign is inverted**: credit added (a PURCHASE) is stored
    negative and credit spent positive, so the current *available* balance in dollars is the
    negated cents / 100 (the docs' own example: ``val "-1000"`` = ``$10.00``). This — the field
    name, the string type, and the inverted sign — was verified live against a real account
    (issue #179), where the ``changes`` ledger summed exactly to ``total``.
    """
    total = data.get("total")
    if not isinstance(total, dict) or "val" not in total:
        raise _BalanceUnavailable("the xAI Management API response carried no balance total.")
    try:
        cents = int(str(total["val"]))
    except (TypeError, ValueError):
        raise _BalanceUnavailable(
            "the xAI Management API returned a non-numeric balance."
        ) from None
    return -cents / 100.0


def _render(balance_usd: float, as_of: datetime) -> str:
    """Format the balance as one model-readable line carrying the figure and an ``as of`` stamp."""
    stamp = as_of.isoformat(timespec="seconds").replace("+00:00", "Z")
    if balance_usd < 0:
        figure = f"-${abs(balance_usd):,.2f} (the account is overdrawn)"
    else:
        figure = f"${balance_usd:,.2f}"
    return f"xAI prepaid credit balance: {figure} USD (as of {stamp})."
