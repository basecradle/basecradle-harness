"""Read the agent's own OpenRouter credit runway — cost self-awareness (issues #425, #853).

The OpenRouter mirror of `_xai_account.py`: same shape, same soft-failure contract, same
locked-profile-safe plumbing — a plain read-only function `Tool` (no platform client, no policy
capability, no shell) that makes authenticated HTTPS GETs and returns figures. An agent holding
an OpenRouter account can see its remaining credit and reason about its runway: throttle,
prioritize cheap experiments, or ask a human to top up *before* it runs dry as a hard API failure.

**Which key it reads with (issue #853).** It used to require a dedicated Management key
(``OPENROUTER_MANAGEMENT_KEY``), on the word of a 2026-08-28 reading that an inference key was
refused on ``/credits``. That reading is stale: on 2026-11-10 an ordinary inference key
(``is_management_key: false``) read both ``GET /credits`` and ``GET /key`` with HTTP 200, on the
global host and on the ``us`` and ``eu`` regional hosts. A Management key can mint and revoke
every key on the account, so requiring one to read a balance handed out far more power than the
read needs. The tool now reads with the first of these that applies:

1. ``OPENROUTER_MANAGEMENT_KEY`` when it is set — an **optional override**, never a requirement,
   kept so an agent provisioned with one keeps reading exactly as before.
2. The brain's own ``AI_API_KEY``, when ``AI_PROVIDER`` is ``openrouter`` (the biller, so the key
   is an OpenRouter key whatever SDK carries it), read at the brain's own host (``AI_BASE_URL``,
   else OpenRouter's).
3. The memory reranker's ``HARNESS_MEMPALACE_RERANK_API_KEY``, which is an OpenRouter key by
   construction (the reranker runs only on OpenRouter), read at the reranker's own host
   (``HARNESS_MEMPALACE_RERANK_BASE_URL``, else OpenRouter's).
4. The describer's ``HARNESS_DESCRIBER_API_KEY``, only when ``HARNESS_DESCRIBER_PROVIDER`` names
   ``openrouter``: a describer riding the brain's stack bills the brain's vendor, which step 2
   already covers when that vendor is OpenRouter. Read at ``AI_BASE_URL`` when the brain bills
   OpenRouter too, because the describer then inherits it, else at OpenRouter's host.

The brain comes before the helpers because it is the agent's main vendor, the account the issue
asked every agent to see. Each key is read **at the host its own calls go to**, so a key kept to
a regional host is never sent to another. Every key here is one the agent already holds and
already sends to OpenRouter; none is a new credential, and none but the optional Management key
can administer anything.

**Why there is still no ``Vendor`` gate** (issue #425). Steps 3 and 4 are the case: an agent
brained by another vendor whose OpenRouter key serves a helper. A ``Vendor("openrouter")`` gate
would self-exclude exactly that agent. The provider check in step 2 is what keeps another vendor's
``AI_API_KEY`` from ever being sent to OpenRouter's host.

**On by default where a key is held** (issue #862). @origin's rule of 2026-11-10: *every agent can
read the available balance at its main LLM vendor, as long as that read exposes no other sensitive
account data, and an agent whose reranker is on OpenRouter reads the OpenRouter balance too.* This
read exposes the account's two credit totals and the calling key's own spend and limit, nothing
else, so it is a benign default rather than an opt-in. Its gate is `OpenRouterKeyHeld`, which asks
`held_key` the same question the tool asks when it runs: an agent holding no OpenRouter key gets no
tool, never one that answers "not configured" on every wake.

**The arithmetic is a subtraction, and it is the whole of it.** ``GET /credits`` returns two
*lifetime cumulative* totals — ``data.total_credits`` (purchased to date) and ``data.total_usage``
(used to date) — as plain positive JSON numbers, and the remaining credit is their difference.
None of the xAI Management API's traps exist here: no cents strings, no inverted sign convention,
no team-UUID discovery, and **no posted-ledger vs. invoice-preview dichotomy**. Do not go looking
for a second surface or invent a degraded tier; there is nothing to fall back *to*, so an unusable
response is simply an unavailable one.

Two consequences of "lifetime cumulative" that are easy to get wrong and are load-bearing here:

- **The context line says "to date", never "this billing cycle".** These totals do not reset at
  cycle close, so cycle wording would be false — and a runway figure the model misreads as a
  *cycle* budget is the same class of defect as xAI's stale posted ledger (issue #384): a number
  that means something other than what it appears to.
- **The difference can legitimately go negative.** Usage can overrun purchased credit, so the
  overdraft callout is kept from the xAI tool rather than assumed impossible.

**The key's own figures** (``GET /key``) sit beside the account's: what the calling key has spent
to date (``usage``) and, when the key has a spending limit, what of it is left
(``limit_remaining``) and how the limit resets (``limit_reset``). A key with a limit can be unable
to pay for a request while the account still holds credit, so a nearly exhausted limit is the
other runway, and it is called out first when it is used up. Both reads are required: a balance
that silently dropped the key's limit would describe a key with room to spend as healthy. They are
not read for a Management key, which spends nothing itself and so has no usage or limit to report.

The figures are **quantized to cents at the parse boundary**, so the subtraction the tool *shows*
is the subtraction it *did*: the two terms and the headline are mutually consistent by
construction, and a model that re-does the arithmetic off the context line lands on the headline.
(``total_usage`` carries sub-cent precision — ``268.464928179`` in a live reading — which is
noise at runway scale and would otherwise let a rounded display contradict an unrounded headline,
or render a four-tenths-of-a-cent overrun as an alarming ``-$0.00`` overdraft.)

Everything fails **soft**: no key, a rejected key, an unusable host, an unreachable endpoint, or an
unexpected response all return a clear ``"unavailable — <reason>"`` string rather than raising,
so a billing check never derails a wake. And it is careful with what it exposes: it never logs or
returns the key, and it never surfaces a response *body* — OpenRouter's 4xx envelope carries an
``error.message`` (which can echo what it was handed) and a ``user_id`` that is account detail,
so only computed figures leave here.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

import httpx

from basecradle_harness._basecradle import base_url_override, normalize_provider
from basecradle_harness._describer import DESCRIBER_API_KEY_VAR, describer_stack_from_env
from basecradle_harness._observability import tool_failed
from basecradle_harness._plugins import ActivationContext, Requirement
from basecradle_harness._rerank import RERANK_API_KEY_VAR, base_url_from_env
from basecradle_harness._secret import Secret, reveal, secret
from basecradle_harness._tools import NO_PARAMETERS, Tool

#: The vendor whose account this tool reads — the `AI_PROVIDER` whose `AI_API_KEY` it may use.
VENDOR = "openrouter"

#: The OpenRouter API root, used for any key that has no host of its own configured. Overridable
#: only for a proxy or a test double, never to reach another vendor.
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

#: The optional account-administration credential that, when set, the tool reads with first.
MANAGEMENT_KEY_VAR = "OPENROUTER_MANAGEMENT_KEY"

#: Per-request HTTP timeout (seconds). A billing read is a quick GET, and a balance check must
#: never block a wake for long, so this is short — a timeout is a ceiling, not a fixed wait.
DEFAULT_TIMEOUT = 15.0

#: How long a fetched figure is reused before re-reading (seconds). Credit moves slowly, and a
#: model reasoning about its runway may check it more than once in a turn; a short cache keeps
#: those from each hitting the billing endpoint. ``0`` disables caching entirely.
DEFAULT_CACHE_TTL = 30.0

#: The ``limit_reset`` periods OpenRouter offers a key's spending limit, rendered as said.
_RESET_PERIODS = frozenset({"daily", "weekly", "monthly"})


class _BalanceUnavailable(Exception):
    """An internal signal carrying a model-readable reason a figure couldn't be read.

    Raised by the request/parse helpers and turned into the tool's ``"unavailable — <reason>"``
    return string by `run`. Its message is always safe to show the model — it names *what* went
    wrong (a status code, a shape problem), never the key or the raw payload.
    """

    def __init__(self, message: str, *, slug: str = "provider_error") -> None:
        super().__init__(message)
        #: The `tool` line's `reason` for this failure (issue #822).
        self.slug = slug


@dataclass(frozen=True)
class _Credential:
    """The key a read is made with, where it came from, and the host it is read at.

    The key is a `Secret`, so this object's `repr`, `vars()` and pickle never carry it (issue
    #599); it is revealed only on the line that builds the request.
    """

    key: Secret
    #: What a message names as the key's source: an env var, or the constructor argument.
    source: str
    #: Whose key it is, in the model's terms (``"your brain's key"``), or ``None`` for an argument.
    whose: str | None
    root: str
    #: The env var the root was read from, or ``None`` when it is OpenRouter's own or an argument.
    root_source: str | None
    #: A Management key spends nothing itself, so ``GET /key`` is not read for it.
    management: bool


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


@dataclass(frozen=True)
class _KeyUsage:
    """The calling key's own figures off ``GET /key``, in dollars and cents."""

    used_usd: float
    #: ``None`` when the key has no spending limit of its own.
    limit_usd: float | None
    #: What the limit still allows; ``None`` exactly when `limit_usd` is.
    limit_remaining_usd: float | None
    #: How the limit resets (``"monthly"``), or ``None`` when it does not reset on a schedule.
    limit_reset: str | None


class OpenRouterAccountBalanceTool(Tool):
    """`openrouter_account_balance` — report the credit remaining on the agent's own OpenRouter
    account, and what the key it reads with has spent.

    A plain read-only `Tool` (no platform client, no policy capability) that calls
    ``GET /api/v1/credits`` and ``GET /api/v1/key`` with an OpenRouter key the agent already holds
    (module docstring: the Management key when set, else the brain's, the reranker's or the
    describer's). The headline is ``total_credits − total_usage``: purchased to date, less used to
    date.

    Unlike its xAI sibling its plugin declares **no `Vendor` requirement**, deliberately (issue
    #425): the ordering case is an agent brained by *another* provider whose OpenRouter key serves
    a helper, and a ``Vendor("openrouter")`` gate would self-exclude exactly that agent. It is a
    benign default since issue #862, gated on `OpenRouterKeyHeld`: present wherever the agent holds
    an OpenRouter key, absent wherever it holds none.

    It degrades to a clear ``"OpenRouter account balance unavailable — <reason>"`` string in every
    failure mode (no key, a rejected key, an unusable host, endpoint unreachable, unexpected
    response) rather than raising. It never logs or returns the key or a response body.

    Args:
        api_key: An OpenRouter API key to read with, whose own usage and limit are reported.
        management_key: A Management key to read with instead; its own figures are not read.
            Either argument given, nothing is read from the environment for the key; with
            neither, the key is resolved from the environment at call time (module docstring).
        base_url: The API root (for a proxy or a test double). ``None`` (the default) reads at
            the host the resolved key's own calls go to, else OpenRouter's.
        timeout: Per-request timeout in seconds.
        cache_ttl: Seconds to reuse a fetched figure before re-reading; ``0`` disables it.
        clock: The monotonic clock the cache measures against (injectable for tests).
    """

    name = "openrouter_account_balance"
    description = (
        "Check the credit remaining on your own OpenRouter account right now, in US dollars — "
        "the credits purchased on the account to date, less what has been used to date — and "
        "what the key you read it with has spent, with how much of that key's own spending limit "
        "is left if it has one. Use it to reason about your runway: throttle or prioritize cheap "
        "work when credit is low, or ask a human to top up before you run dry mid-task. Takes no "
        "arguments and can see only your own account and key, nothing else."
    )
    parameters = NO_PARAMETERS

    def __init__(
        self,
        *,
        api_key: str | None = None,
        management_key: str | None = None,
        base_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        cache_ttl: float = DEFAULT_CACHE_TTL,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._api_key = secret(api_key)  # never a plain attribute (issue #599)
        self._management_key = secret(management_key)
        self._base_url = base_url
        self._timeout = timeout
        self._cache_ttl = cache_ttl
        self._clock = clock
        # (monotonic expiry, a digest of the key it was read with, the root, the source, rendered
        # text). The key, root and source are part of the entry because all three are resolved at
        # *call* time: without them, a figure read for one account — with that account's `as of`
        # stamp — is served as another's for the rest of the TTL after a credential is repointed.
        # A digest, never the key: a plain attribute holding it is emitted by `vars()` and by every
        # crash reporter that expands locals, the leak `Secret` exists to close (issue #599).
        self._cached: tuple[float, bytes, str, str, str] | None = None

    def run(self) -> str:
        """Report the credit remaining, or a clear reason it is unavailable."""
        credential = self._credential()
        if credential is None:
            return tool_failed(
                "OpenRouter account balance unavailable — no OpenRouter key is configured for "
                f"you. This tool reads with {MANAGEMENT_KEY_VAR} when it is set, else your "
                f"brain's AI_API_KEY when AI_PROVIDER is openrouter (it is {_active_provider()!r} "
                f"here), else your memory reranker's {RERANK_API_KEY_VAR}, else your describer's "
                f"{DESCRIBER_API_KEY_VAR} when HARNESS_DESCRIBER_PROVIDER is openrouter; none of "
                "those is available.",
                reason="config:missing_api_key",
            )

        key = credential.key.reveal()
        digest = hashlib.sha256(key.encode("utf-8", "surrogatepass")).digest()
        cached = self._cached_balance(digest, credential)
        if cached is not None:
            return cached
        if any(ord(char) < 32 or ord(char) == 127 for char in key):
            # Refused before any client exists, never left to the transport: h11 rejects such a
            # header with an error whose text quotes the whole header, key included, so the
            # failure itself would carry the key into the tool's answer and the log.
            return tool_failed(
                f"OpenRouter account balance unavailable — {credential.source} contains a control "
                "character (a line break or a tab, picked up when it was pasted), so it cannot be "
                "sent as a header.",
                reason="config:auth",
            )
        if not _usable_root(credential.root):
            return _unusable_root(credential)

        try:
            # The client is built *inside* the guard, not outside it: httpx encodes a header
            # value as ASCII at construction time, so a key carrying a smart quote, a
            # non-breaking space or an accented character — the ordinary consequence of pasting
            # a credential — raises `UnicodeEncodeError` before any request exists to fail. A
            # misconfigured credential is the case this tool most owes a soft answer to.
            with httpx.Client(
                headers={"Authorization": f"Bearer {key}"}, timeout=self._timeout
            ) as client:
                credits = _parse_credits(
                    self._get(client, credential, "/credits", "read the credits remaining")
                )
                usage = None
                if not credential.management:
                    usage = _parse_key(
                        self._get(client, credential, "/key", "read this key's own usage")
                    )
        except httpx.InvalidURL:
            return _unusable_root(credential)
        except UnicodeEncodeError:
            return tool_failed(
                f"OpenRouter account balance unavailable — {credential.source} contains a "
                "non-ASCII character, so it cannot be sent as a header; check for a smart quote "
                "or a non-breaking space picked up when it was pasted.",
                reason="config:auth",
            )
        except _BalanceUnavailable as exc:
            return tool_failed(f"OpenRouter account balance unavailable — {exc}", reason=exc.slug)

        text = _render(credits, usage, credential, self._now_utc())
        if self._cache_ttl > 0:
            entry = (self._clock() + self._cache_ttl, digest, credential.root, credential.source)
            self._cached = (*entry, text)
        return text

    # --- which key, at which host ----------------------------------------------

    def _credential(self) -> _Credential | None:
        """The key this call reads with, in the module docstring's order, or ``None`` if none.

        Read at call time, so a credential set or repointed after the tool was built is the one
        used. A key handed to the constructor wins, and then nothing is read from the environment.
        """
        argument_root = (self._base_url or DEFAULT_BASE_URL).rstrip("/")
        for value, source, management in (
            (self._management_key, "the management_key argument", True),
            (self._api_key, "the api_key argument", False),
        ):
            key = reveal(value)
            if key:
                return _Credential(secret(key), source, None, argument_root, None, management)

        held = held_key(os.environ, _active_provider())
        if held is None:
            return None
        var, key = held
        if var == MANAGEMENT_KEY_VAR:
            return self._from_env(key, var, "a Management key", None, None, True)
        if var == "AI_API_KEY":
            root = base_url_override()
            return self._from_env(key, var, "your brain's key", root, "AI_BASE_URL")
        if var == RERANK_API_KEY_VAR:
            root = base_url_from_env()
            where = "HARNESS_MEMPALACE_RERANK_BASE_URL"
            return self._from_env(key, var, "your memory reranker's key", root, where)
        # A describer naming the brain's own vendor inherits the brain's host (`_describer`'s
        # `inherit_endpoint`), so a regional `AI_BASE_URL` keeps its key regional here too.
        root, where = (base_url_override(), "AI_BASE_URL")
        if _active_provider() != VENDOR:
            root, where = None, None
        return self._from_env(key, var, "your describer's key", root, where)

    def _from_env(
        self,
        key: str,
        source: str,
        whose: str,
        root: str | None,
        root_var: str | None,
        management: bool = False,
    ) -> _Credential:
        """A credential read from the environment, at its own host unless `base_url` was given."""
        if self._base_url:
            root, root_var = self._base_url, None
        elif not root:
            root, root_var = DEFAULT_BASE_URL, None
        return _Credential(secret(key), source, whose, root.rstrip("/"), root_var, management)

    # --- the HTTP calls -------------------------------------------------------

    def _get(
        self, client: httpx.Client, credential: _Credential, path: str, action: str
    ) -> dict[str, Any]:
        """GET an API path and decode it, mapping every failure to a soft reason.

        Never surfaces the response *body*: OpenRouter's error envelope carries an
        ``error.message`` that can echo what it was handed and a ``user_id``, and the contract is
        that only computed figures ever leave this tool.
        """
        try:
            response = client.get(f"{credential.root}{path}")
        except httpx.TimeoutException:
            raise _BalanceUnavailable(
                f"the OpenRouter API did not answer in time trying to {action}.", slug="timeout"
            ) from None
        except httpx.RequestError as exc:
            # The error's *type*, never its text: a transport error can quote the request it
            # failed on, headers included (h11's illegal-header error carries the whole
            # `Authorization` value), and the contract is that the key never leaves this tool.
            raise _BalanceUnavailable(
                f"couldn't reach the OpenRouter API ({type(exc).__name__}).", slug="transport"
            ) from None

        status = response.status_code
        if status in (401, 403):
            raise _BalanceUnavailable(
                f"OpenRouter rejected the key in {credential.source} (HTTP {status}) trying to "
                f"{action}. Check that {credential.source} holds a current OpenRouter key.",
                slug="config:auth",
            )
        if not 200 <= status < 300:
            # Named by `docs/telemetry.md` → Fault Slugs, the same words on every producer.
            slug = _STATUS_SLUGS.get(status) or ("server_error" if status >= 500 else "api_error")
            raise _BalanceUnavailable(
                f"OpenRouter returned HTTP {status} trying to {action}.", slug=slug
            )
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

    def _cached_balance(self, digest: bytes, credential: _Credential) -> str | None:
        """The cached text if it is still fresh **and** was read with this key, here."""
        if self._cached is None:
            return None
        expiry, cached_digest, root, source, text = self._cached
        same_read = root == credential.root and source == credential.source
        if hmac.compare_digest(cached_digest, digest) and same_read and self._clock() < expiry:
            return text
        return None

    def _now_utc(self) -> datetime:
        """Wall-clock now, in UTC — the ``as of`` stamp on a freshly-read figure."""
        return datetime.now(UTC)


#: The statuses `docs/telemetry.md` → Fault Slugs names by the status alone, beyond ``401``/``403``,
#: which this tool words for itself. Any other ``5xx`` is ``server_error`` and any other status (a
#: ``3xx`` httpx did not follow included) is ``api_error``, as that table rules.
_STATUS_SLUGS = {
    402: "config:billing",
    404: "config:model_not_found",
    413: "payload_too_large",
    429: "rate_limited",
    504: "timeout",
}


def _usable_root(root: str) -> bool:
    """Whether `root` is an ``http(s)`` URL with a host, the only kind a read can go to.

    Checked before any request, because a schemeless value (``us.openrouter.ai/api/v1``) fails in
    the transport as ``UnsupportedProtocol``, which would read as a network failure that retrying
    can fix rather than the configuration fault it is (the reranker judges the same value so).
    """
    try:
        parts = urlsplit(root)
    except ValueError:
        return False
    return parts.scheme in ("http", "https") and bool(parts.hostname)


def _unusable_root(credential: _Credential) -> str:
    """The soft answer for a host the read cannot go to, naming where that host came from."""
    where = credential.root_source or "the base_url argument"
    return tool_failed(
        f"OpenRouter account balance unavailable — {where} is not a usable URL, so there is no "
        "host to read the account at.",
        reason="config:base_url",
    )


def _value(env: Mapping[str, str], var: str) -> str | None:
    """A variable's value in `env`, stripped, or ``None`` when it is unset or blank."""
    return (env.get(var) or "").strip() or None


#: Every variable an OpenRouter key may be read from, in the order `held_key` reads them.
KEY_VARS = (MANAGEMENT_KEY_VAR, "AI_API_KEY", RERANK_API_KEY_VAR, DESCRIBER_API_KEY_VAR)


def held_key(env: Mapping[str, str], provider: str) -> tuple[str, str] | None:
    """``(variable, key)`` for the OpenRouter key this tool reads with, or ``None`` if none is held.

    The module docstring's order, and the **one** place it is written down: the tool reads its key
    through this (`OpenRouterAccountBalanceTool._credential`), and its plugin's activation gate asks
    it the same question (`OpenRouterKeyHeld`), so the tool can never be offered to an agent it
    would then answer "no OpenRouter key is configured", nor withheld from one whose key it would
    have read. `provider` is ``AI_PROVIDER`` normalized as a wake reads it, because the brain's
    ``AI_API_KEY`` is an OpenRouter key only when OpenRouter bills the brain.
    """
    describer_on_openrouter = _normalized(describer_stack_from_env(env)[0]) == VENDOR
    for var, applies in (
        (MANAGEMENT_KEY_VAR, True),
        ("AI_API_KEY", provider == VENDOR),
        (RERANK_API_KEY_VAR, True),
        (DESCRIBER_API_KEY_VAR, describer_on_openrouter),
    ):
        key = _value(env, var)
        if key and applies:
            return var, key
    return None


@dataclass(frozen=True)
class OpenRouterKeyHeld(Requirement):
    """Met iff the agent already holds an OpenRouter key this tool would read with (issue #862).

    The activation gate of ``openrouter_account_balance``, which is on by default since @origin's
    rule of 2026-11-10: *every agent can read the available balance at its main LLM vendor, as long
    as that read exposes no other sensitive account data, and an agent whose reranker is on
    OpenRouter reads the OpenRouter balance too.* The tool is offered exactly where it can work: an
    agent holding no OpenRouter key gets no tool, never a tool that answers "not configured" on
    every wake. The precedence is `held_key`, so the gate and the tool cannot disagree.

    Deliberately **not** a `Vendor` gate (issue #425): an agent brained by another vendor whose
    OpenRouter key serves its reranker or describer is the case it exists for. And deliberately
    **not** an `EnvSet`: any one of four keys meets it, so it requires none of them. It names all
    four as `credential_vars` (what ``basecradle-harness-resolve`` assumes present or absent) but
    contributes nothing to `plugin_env_dependencies`, so a healthy agent reading with its brain's
    key never shows three unset variables as gaps in ``tool_env`` (the issue #853 reasoning).
    """

    def met(self, ctx: ActivationContext) -> bool:
        return held_key(ctx.env, ctx.provider) is not None

    @property
    def reason(self) -> str:
        return (
            f"needs an OpenRouter key the agent already holds ({MANAGEMENT_KEY_VAR}; AI_API_KEY "
            f"under AI_PROVIDER=openrouter; {RERANK_API_KEY_VAR}; or {DESCRIBER_API_KEY_VAR} under "
            "HARNESS_DESCRIBER_PROVIDER=openrouter)"
        )

    @property
    def credential_vars(self) -> tuple[str, ...]:
        return KEY_VARS


def _normalized(raw: str | None) -> str:
    """A provider slug as the wake reads it, or the raw value when it is not even a slug."""
    try:
        return normalize_provider(raw)
    except ValueError:
        return (raw or "").strip()


def _active_provider() -> str:
    """``AI_PROVIDER`` as the wake reads it."""
    return _normalized(os.environ.get("AI_PROVIDER"))


def _parse_credits(data: dict[str, Any]) -> _Credits:
    """The two lifetime totals out of ``GET /credits``.

    Shape: ``{"data": {"total_credits": 500.0, "total_usage": 123.456789}}`` — plain positive
    JSON numbers of USD, verified live 2026-08-28 and again 2026-11-10. The live readings returned
    ``total_credits`` as a bare integer (``375``, ``475``), so an integer is as valid here as a
    float; ``bool`` is not, despite being an `int` subclass in Python.

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


def _parse_key(data: dict[str, Any]) -> _KeyUsage:
    """The calling key's own figures out of ``GET /key``.

    Shape (verified live 2026-11-10, abridged): ``{"data": {"usage": 1.047832516, "limit": null,
    "limit_remaining": null, "limit_reset": null, …}}``, every figure plain USD. ``usage`` is
    required. ``limit`` must be **present**, as a number or ``null``: the vendor states ``null``
    for "no limit", so a missing field is not that statement, and reading it as one would report
    a key with a limit as having none. When a limit is set, ``limit_remaining`` is required, and
    ``limit_reset`` is a period or ``null``. Quantized to cents like the account's figures.
    """
    inner = data.get("data")
    if not isinstance(inner, dict):
        raise _BalanceUnavailable("OpenRouter's key response carried no data object.")
    used = _usd(inner, "usage")
    if used is None:
        raise _BalanceUnavailable("OpenRouter's key response carried no usable usage figure.")
    if "limit" not in inner or (inner["limit"] is not None and _usd(inner, "limit") is None):
        raise _BalanceUnavailable("OpenRouter's key response carried no usable limit figure.")
    limit = _usd(inner, "limit")
    if limit is None:
        return _KeyUsage(used_usd=used, limit_usd=None, limit_remaining_usd=None, limit_reset=None)
    remaining = _usd(inner, "limit_remaining")
    reset = inner.get("limit_reset")
    if remaining is None or not (reset is None or isinstance(reset, str)):
        missing = "limit_remaining" if remaining is None else "limit_reset"
        raise _BalanceUnavailable(
            f"OpenRouter's key response carried a spending limit but no usable {missing}."
        )
    return _KeyUsage(
        used_usd=used, limit_usd=limit, limit_remaining_usd=remaining, limit_reset=reset
    )


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


def _key_line(usage: _KeyUsage | None, credential: _Credential) -> str:
    """What the key the figures were read with has spent, and what its own limit leaves."""
    named = credential.source + (f", {credential.whose}" if credential.whose else "")
    if usage is None:
        return (
            f"Read with {named}, which spends nothing itself and so has no usage or spending "
            "limit of its own."
        )
    spent = f"The key this was read with ({named}) has spent {_dollars(usage.used_usd)} to date"
    if usage.limit_usd is None or usage.limit_remaining_usd is None:
        return f"{spent} and has no spending limit of its own."
    if usage.limit_reset is None:
        reset = "and does not reset on a schedule"
    elif usage.limit_reset in _RESET_PERIODS:
        reset = f"and resets {usage.limit_reset}"
    else:
        reset = "on a reset period this tool does not recognize"
    return (
        f"{spent}; its spending limit of {_dollars(usage.limit_usd)} allows "
        f"{_dollars(usage.limit_remaining_usd)} more, {reset}."
    )


def _render(
    credits: _Credits, usage: _KeyUsage | None, credential: _Credential, as_of: datetime
) -> str:
    """The headline figure, with the subtraction that produced it and the key's figures beneath.

    The headline shape is **uniform** — ``<label>: <$figure> USD (as of <stamp>).`` — so the
    figure is always in the same place and a reader (or a regex, or the model, or the NOC's live
    probe) never has to parse around a parenthetical that appears only sometimes; an overdraft,
    or a key whose own limit is used up, is called out in the body instead, where it leads so it
    is not buried.

    The subtraction is **shown**, not merely performed: an agent that meets ``total_credits``
    elsewhere — OpenRouter's own dashboard, another client — can see which number that is and why
    it is the larger one. The wording is "to date" on both terms because these are lifetime
    cumulative totals that never reset; calling either one a billing cycle's would be false.
    """
    context = ["The account is overdrawn."] if credits.remaining_usd < 0 else []
    left = usage.limit_remaining_usd if usage is not None else None
    if left is not None and left <= 0:
        context.append(
            "This key's spending limit is used up, so no request made with it can be paid for "
            "until the limit resets or is raised."
        )
    context.append(
        f"Live figure — {_dollars(credits.purchased_usd)} of credits purchased to date less the "
        f"{_dollars(credits.used_usd)} used to date."
    )
    context.append(_key_line(usage, credential))
    headline = (
        f"OpenRouter credits remaining: {_dollars(credits.remaining_usd)} USD "
        f"(as of {_stamp(as_of)})."
    )
    return f"{headline}\n" + " ".join(context)
