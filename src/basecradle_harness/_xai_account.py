"""Read the agent's own xAI credit runway — cost self-awareness (issues #179, #384, #388).

An xAI persona whose charter treats capital as a first-class concern can see its remaining
runway and reason about it — throttle, prioritize cheap experiments, or ask a human to top up
*before* it runs dry as a hard API failure. This is the tool that gives it that sense.

It talks to the xAI **Management API** (`management-api.x.ai`), a billing/account surface
distinct from the inference endpoint (`api.x.ai`), with its own dedicated credential — a
read-only **Management Key** (`XAI_MANAGEMENT_KEY`), never the agent's inference `AI_API_KEY`.
So it is a plain, read-only function `Tool` (no platform client, no policy capability, no shell)
that makes authenticated HTTPS GETs and returns a figure — the same locked-profile-safe shape
as `web_fetch`.

**The posted ledger is not the runway, and that is the whole point of this tool (issue #384).**
`…/prepaid/balance` returns a *settled* ledger: its SPEND rows are keyed by billing period and
land at **cycle close**, not continuously. So mid-cycle it reports credit that has in fact
already been consumed, and the gap grows as the cycle progresses — measured live on 2026-08-02,
the ledger said **$517.49** while the Console showed **$47.42** remaining against **$469.51** of
usage. That is not a parse or a sign bug; it is the wrong *question*. A runway instrument that
silently answers it is worse than no instrument, because an agent reads the number as spendable
and keeps spending. The live figure lives on the **postpaid invoice preview**
(`…/postpaid/invoice/preview`). **Do not "simplify" this back to the ledger alone.** The ledger
is still read, but only as *context*, and it is never presented as remaining credit — when the
preview is unavailable the tool says so in the same breath as the ledger total and calls it an
upper bound.

**The live figure is a *subtraction*, and reading one field of the preview was the same defect
one surface in (issue #388).** `coreInvoice.prepaidCredits` is not "credit left to spend": it is
the prepaid credit this cycle is drawing *against*, before netting what it has already drawn, and
`coreInvoice.prepaidCreditsUsed` is that draw. The Console's **Credits remaining** is the
difference. Reporting the first field alone overstated runway roughly threefold on a live account
— 2026-08-02 21:14 UTC, `prepaidCredits` **$168.47** against a Console showing **$52.14**, where
$168.47 − $116.66 drawn = **$51.81** — so getting the surface right in #384 fixed the *question*
and left the *arithmetic* wrong. Both fields are therefore required: a preview carrying only
`prepaidCredits` is not a live reading at all and is treated as an unavailable one, never as
remaining credit. And `totalWithCorr` is **not** a stand-in for the draw — it is the cycle's
*total* spend, including whatever ran past prepaid onto the postpaid invoice, so subtracting it
renders a merely-exhausted account as a phantom overdraft.

Two things the live API taught us that the naive sketch got wrong (both verified against a real
account, issue #179):

- **A figure lives at `<field>.val` as a string of USD *cents*** — not `.amount`/`.value`. And
  the **sign convention is inverted**: *credit* is stored negative and *spend* positive, so a
  credit figure in dollars is the *negated* cents / 100 (the docs' own example: `val "-1000"` =
  `$10.00`). Getting this wrong reports a healthy positive balance as a negative one. The same
  convention governs the preview: `prepaidCredits` and `prepaidCreditsUsed` are credit-signed
  (negative), while `totalWithCorr` is a spend figure and is already positive.
- **The team path segment is a UUID, not the literal `"default"`** — that 400s (`Invalid uuid`).
  The key knows its own team, so the tool *discovers* it from the management-key validation
  endpoint; `XAI_TEAM_ID` is an optional override, not a required (and misleading) default.

Everything fails **soft**: a missing key, the wrong scope, an unreachable endpoint, or an
unexpected response all return a clear "unavailable — <reason>" string rather than raising, so a
billing check never derails a wake. And it is careful with what it exposes: it never logs or
returns the key, and it never returns a raw billing payload (whose `changes` ledger carries
purchase/invoice history) — only the computed figures the agent needs.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
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
class _Cycle:
    """The live mid-cycle picture, read off the postpaid invoice preview.

    `remaining_usd` is the answer the tool exists to give, and it is a **subtraction** (issue
    #388): the prepaid credit this cycle draws against, less what it has drawn so far. It is
    *derived* rather than stored, so the headline figure and the two terms behind it can never
    disagree. `usage_usd` is context and optional — the cycle's *total* spend, which runs past
    the prepaid draw once prepaid is exhausted and is therefore never a stand-in for it.
    """

    prepaid_usd: float
    prepaid_used_usd: float
    usage_usd: float | None

    @property
    def remaining_usd(self) -> float:
        """Prepaid credit net of this cycle's draw — the Console's *Credits remaining*."""
        return self.prepaid_usd - self.prepaid_used_usd


class XaiAccountBalanceTool(Tool):
    """`xai_account_balance` — report the credit remaining on this agent's own xAI account.

    A plain read-only `Tool` (no platform client, no policy capability) that calls the xAI
    Management API with a dedicated `XAI_MANAGEMENT_KEY` (read-only billing scope), so an xAI
    agent can see its remaining credit and reason about its runway. Its plugin gates it to the
    xAI provider (`Vendor("xai")`) and marks it opt-in like every powerful tool.

    The figure it leads with is the **live** one — the postpaid invoice preview's prepaid credit
    *less this cycle's draw against it* (issue #388), which is what the Console calls **Credits
    remaining**. It is never the posted prepaid ledger, which settles at cycle close and overstates
    runway mid-cycle (issue #384), and never the preview's undrawn `prepaidCredits` alone. The
    ledger is read as context and, if the preview is unavailable, reported *as a labelled upper
    bound* rather than as remaining credit.

    It degrades to a clear ``"xAI account balance unavailable — <reason>"`` string in every
    failure mode (no key, wrong scope, endpoint unreachable, unexpected response) rather than
    raising. It never logs or returns the key or a raw billing payload — only computed figures.

    Args:
        management_key: The Management Key. ``None`` (the default) reads `XAI_MANAGEMENT_KEY`
            from the environment at call time.
        team_id: The team UUID. ``None`` reads `XAI_TEAM_ID`, and if that is unset too the team
            is discovered from the key itself (and cached on the instance).
        base_url: The Management API root (for a proxy or a test double).
        timeout: Per-request timeout in seconds.
        cache_ttl: Seconds to reuse a fetched figure before re-reading; ``0`` disables it.
        clock: The monotonic clock the cache measures against (injectable for tests).
    """

    name = "xai_account_balance"
    description = (
        "Check the credit remaining on your own xAI account right now, in US dollars — the live "
        "figure, net of what this billing cycle has already drawn from your prepaid credit (which "
        "the posted prepaid ledger does not yet reflect, and so overstates). Use it to reason "
        "about your runway: "
        "throttle or prioritize cheap work when credit is low, or ask a human to top up before "
        "you run dry mid-task. Takes no arguments and can see only your own account, nothing "
        "else. (When an X Developer account is linked, X API credit purchases grant free xAI "
        "credits, so this figure can reflect X-side spend rewards too.)"
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
        self._cached: tuple[float, str] | None = None  # (monotonic expiry, rendered text)

    def run(self) -> str:
        """Report the live credit remaining, or a clear reason it is unavailable."""
        key = self._management_key or os.environ.get("XAI_MANAGEMENT_KEY")
        if not key:
            return (
                "xAI account balance unavailable — XAI_MANAGEMENT_KEY is not configured. Set it "
                "to a read-only xAI Management Key (console.x.ai → Settings → Management Keys)."
            )

        cached = self._cached_balance()
        if cached is not None:
            return cached

        headers = {"Authorization": f"Bearer {key}"}
        with httpx.Client(headers=headers, timeout=self._timeout) as client:
            # The team is a hard precondition — neither figure is reachable without it.
            try:
                team = self._team(client)
            except _BalanceUnavailable as exc:
                return f"xAI account balance unavailable — {exc}"

            # The two figures are *alternatives*, not a chain: the preview is the answer, the
            # ledger is context, and either can fail without taking the other down with it.
            try:
                cycle: _Cycle | None = self._cycle(client, team)
                no_cycle = ""
            except _BalanceUnavailable as exc:
                cycle, no_cycle = None, str(exc)
            try:
                posted: float | None = self._posted(client, team)
            except _BalanceUnavailable:
                posted = None

        as_of = self._now_utc()
        if cycle is not None:
            text = _render_live(cycle, posted, as_of)
        elif posted is not None:
            text = _render_posted_only(posted, no_cycle, as_of)
        else:
            return f"xAI account balance unavailable — {no_cycle}"

        if self._cache_ttl > 0:
            self._cached = (self._clock() + self._cache_ttl, text)
        return text

    # --- the three HTTP calls -------------------------------------------------

    def _team(self, client: httpx.Client) -> str:
        """The team UUID: an explicit override, else discovered from the key (then cached).

        The billing endpoints' path segment is a team *UUID* — the literal ``"default"`` 400s
        (``Invalid uuid``). The management-key validation endpoint reports the team the key acts
        for, so a correctly-scoped key needs no `XAI_TEAM_ID` at all; the override just skips the
        discovery call.
        """
        override = self._team_id or os.environ.get("XAI_TEAM_ID")
        if override:
            return override
        if self._resolved_team is not None:
            return self._resolved_team

        data = self._get(client, "/auth/management-keys/validation", "validate the management key")
        team = data.get("teamId")
        if not isinstance(team, str) or not team:
            raise _BalanceUnavailable(
                "the xAI Management API did not report a team for this key. Set XAI_TEAM_ID to "
                "your team UUID to bypass discovery."
            )
        self._resolved_team = team
        return team

    def _cycle(self, client: httpx.Client, team: str) -> _Cycle:
        """The live mid-cycle figures — the answer, read off the postpaid invoice preview."""
        data = self._get(
            client,
            f"/v1/billing/teams/{team}/postpaid/invoice/preview",
            "read the live credits remaining",
        )
        return _parse_cycle(data)

    def _posted(self, client: httpx.Client, team: str) -> float:
        """The *posted* prepaid ledger total in USD — context only, never the runway figure."""
        data = self._get(
            client,
            f"/v1/billing/teams/{team}/prepaid/balance",
            "read the posted prepaid ledger",
        )
        return _parse_posted_usd(data)

    def _get(self, client: httpx.Client, path: str, action: str) -> dict[str, Any]:
        """GET a Management API path and decode it, mapping every failure to a soft reason.

        Never surfaces the response *body*: a 4xx billing body can echo account detail, and the
        contract is that only computed figures ever leave this tool.
        """
        try:
            response = client.get(f"{self._base_url}{path}")
        except httpx.RequestError as exc:
            raise _BalanceUnavailable(f"couldn't reach the xAI Management API ({exc}).") from None

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
        """The cached text if still fresh, else ``None``."""
        if self._cached is None:
            return None
        expiry, text = self._cached
        if self._clock() < expiry:
            return text
        return None

    def _now_utc(self) -> datetime:
        """Wall-clock now, in UTC — the ``as of`` stamp on a freshly-read figure."""
        return datetime.now(timezone.utc)


def _parse_cycle(data: dict[str, Any]) -> _Cycle:
    """The live mid-cycle figures from the postpaid invoice preview.

    Shape: ``{"coreInvoice": {"prepaidCredits": {"val": "-16847"}, "prepaidCreditsUsed": {"val":
    "-11666"}, "totalWithCorr": {"val": "11666"}}}``. Credit figures are stored **negative** (the
    module-wide convention) and are negated back to dollars; ``totalWithCorr`` is a *spend* figure
    and is already positive.

    ``prepaidCredits`` **and** ``prepaidCreditsUsed`` are both required, because the answer is
    their difference (issue #388) — the first alone is the credit the cycle draws *against*, not
    what is left of it. So a preview carrying only the first is not a live reading at all: it
    raises, and the caller falls back to the posted ledger *labelled an upper bound*, which is the
    single place this tool is permitted to show a figure that is not the runway. Returning
    ``prepaidCredits`` as remaining is the defect itself, and it is the shape a "tolerant" parse
    reintroduces.

    Both are **negated, never ``abs()``d:** an absolute value would render a positive (overdrawn)
    reading as healthy available credit, silently inverting the one condition this tool exists to
    warn about.
    """
    core = data.get("coreInvoice")
    if not isinstance(core, dict):
        raise _BalanceUnavailable(
            "the xAI Management API's invoice preview carried no coreInvoice."
        )
    prepaid_cents = _cents(core, "prepaidCredits")
    used_cents = _cents(core, "prepaidCreditsUsed")
    if prepaid_cents is None or used_cents is None:
        missing = "prepaidCredits" if prepaid_cents is None else "prepaidCreditsUsed"
        raise _BalanceUnavailable(
            f"the xAI Management API's invoice preview carried no usable {missing} figure; the "
            "live remaining credit is prepaidCredits minus prepaidCreditsUsed, so both are needed."
        )
    usage_cents = _cents(core, "totalWithCorr")
    return _Cycle(
        prepaid_usd=-prepaid_cents / 100.0,
        prepaid_used_usd=-used_cents / 100.0,
        usage_usd=None if usage_cents is None else usage_cents / 100.0,
    )


def _cents(holder: dict[str, Any], field: str) -> int | None:
    """The integer cents at ``holder[field]["val"]``, or ``None`` if absent or unparseable."""
    figure = holder.get(field)
    if not isinstance(figure, dict) or "val" not in figure:
        return None
    try:
        return int(str(figure["val"]))
    except (TypeError, ValueError):
        return None


def _parse_posted_usd(data: dict[str, Any]) -> float:
    """The **posted** prepaid ledger total in USD — a settled figure, not the runway.

    The endpoint returns ``{"total": {"val": "<cents>"}, "changes": [...]}`` where ``val`` is a
    string of USD cents in the inverted (credit-negative) convention, so the total in dollars is
    the negated cents / 100. Its SPEND rows settle at **cycle close**, which is exactly why this
    figure is context and never the answer — see the module docstring and issue #384.
    """
    total = data.get("total")
    if not isinstance(total, dict) or "val" not in total:
        raise _BalanceUnavailable("the xAI Management API response carried no ledger total.")
    try:
        cents = int(str(total["val"]))
    except (TypeError, ValueError):
        raise _BalanceUnavailable(
            "the xAI Management API returned a non-numeric ledger total."
        ) from None
    return -cents / 100.0


def _dollars(usd: float) -> str:
    """A signed USD figure — ``$42.50`` / ``-$5.00``."""
    return f"-${abs(usd):,.2f}" if usd < 0 else f"${usd:,.2f}"


def _stamp(as_of: datetime) -> str:
    """The ``as of`` timestamp, ISO-8601 to the second with a ``Z`` suffix."""
    return as_of.isoformat(timespec="seconds").replace("+00:00", "Z")


def _headline(label: str, usd: float, as_of: datetime) -> str:
    """One uniform headline shape: label, signed figure, ``USD``, stamp.

    Deliberately *uniform* — an overdraft is called out in the body rather than spliced into the
    headline, so the figure is always in the same place and a reader (or a regex, or the model)
    never has to parse around a parenthetical that appears only sometimes.
    """
    return f"{label}: {_dollars(usd)} USD (as of {_stamp(as_of)})."


def _overdraft(usd: float) -> list[str]:
    """The overdraft sentence, when there is one — leading the context so it is not buried."""
    return ["The account is overdrawn."] if usd < 0 else []


def _render_live(cycle: _Cycle, posted_usd: float | None, as_of: datetime) -> str:
    """The good case: the live remaining figure, with the burn and the ledger behind it.

    The subtraction is **shown**, not merely performed (issue #388): an agent that meets
    `prepaidCredits` elsewhere — the Console's own invoice preview, another client — can see which
    number that is and why it is the larger one. The posted ledger is included *labelled* for the
    same reason: every bigger figure this account can show is named here, so none of them can pass
    itself off as spendable.
    """
    context = _overdraft(cycle.remaining_usd)
    context.append(
        f"Live figure — {_dollars(cycle.prepaid_usd)} of prepaid credit less the "
        f"{_dollars(cycle.prepaid_used_usd)} this billing cycle's usage has drawn from it."
    )
    if cycle.usage_usd is not None:
        context.append(f"This cycle: {_dollars(cycle.usage_usd)} used in total.")
    if posted_usd is not None:
        context.append(
            f"Posted prepaid ledger: {_dollars(posted_usd)} — that total settles only at cycle "
            "close, so mid-cycle it overstates runway; it is not what you have left to spend."
        )
    headline = _headline("xAI credits remaining", cycle.remaining_usd, as_of)
    return f"{headline}\n" + " ".join(context)


def _render_posted_only(posted_usd: float, reason: str, as_of: datetime) -> str:
    """The degraded case: the ledger total, explicitly *not* offered as remaining credit.

    The one thing this must never do is let a posted total pass as a live one (issue #384), so
    the disclaimer leads the second line rather than trailing it, and the label on the figure
    itself says "posted prepaid ledger", never "remaining" or "available".
    """
    context = [
        (
            f"This is NOT your remaining credit — the live figure was unavailable: {reason} The "
            "posted ledger settles only at cycle close, so mid-cycle it overstates runway by "
            "roughly this cycle's unbilled usage. Treat it as an upper bound, not a budget."
        ),
        *_overdraft(posted_usd),
    ]
    headline = _headline("xAI posted prepaid ledger total", posted_usd, as_of)
    return f"{headline}\n" + " ".join(context)
