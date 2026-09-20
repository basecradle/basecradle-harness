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
    """The call failed in transport — DNS, TCP, TLS, a connect timeout, or a **read** timeout.

    Transient, and therefore retried on every model call the harness makes (issue #545): a
    transport that failed once usually succeeds when re-issued, and behind a router the re-issue is
    re-routed besides.

    **It covers two genuinely different events and the adapters cannot tell them apart**, which is
    why the taxonomy reads the distinction off the ``__cause__`` instead
    (`basecradle_harness._retry.connection_reason` → ``transport`` or ``timeout``). A *connect*
    failure means nothing reached the model. A *read* timeout means the request was accepted and
    the answer did not come back in time — the model may have run, and may have been billed. Both
    are safe to retry, because neither can have dispatched a tool: the harness acts only on a
    parsed response.
    """


class ProviderToolSchemaError(ProviderError):
    """The provider refused **one tool's** JSON Schema, naming it (issue #496).

    Deliberately its own class, and deliberately **not** a `ProviderRequestError`: nothing about the
    peer's content is at fault, so this must never be reported to a timeline as a permanent property
    of what someone sent. It is a fault of the *offer* — a schema a vendor's validator will not take
    — and the remedy is entirely local: stop offering that tool.

    So an adapter raises it, and its **own `chat` catches it**, drops the named tool, and re-issues
    the identical turn with the rest (`XaiSdkProvider._sample`). It reaches the engine only in the
    residual case where the vendor named no tool this call actually offered — and then it behaves
    exactly like the plain `ProviderError` it descends from: it propagates, the wake fails visibly,
    and the peer's message stays re-drivable rather than being marked answered.

    `tool_name` is the tool the vendor named (``None`` when it named none); `reason` is the vendor's
    own words for why, kept so the WARNING that records the drop quotes the authority on it rather
    than the harness's guess.
    """

    def __init__(self, message: str, *, tool_name: str | None = None, reason: str = "") -> None:
        super().__init__(message)
        self.tool_name = tool_name
        self.reason = reason


class ProviderAPIError(ProviderError):
    """The provider returned an error status.

    `status_code` is the HTTP status; `body` is the raw response text, kept for
    debugging since provider error schemas are not standardized.

    **The remaining four are the vendor's own account of the failure** (issue #506), parsed once by
    whichever adapter understands its vendor's error shape and read as plain attributes everywhere
    else — so a retry policy and a fallback line quote the authority on the fault rather than
    re-deriving it from JSON at three call sites. Every one defaults to ``None``, which is what an
    adapter with nothing to say carries, and what `kv` then omits from the line:

    - `retry_after` — the seconds a ``Retry-After`` header hinted. It lives here rather than on
      `ProviderRateLimitError` alone because a 5xx may carry one too, and a retry that ignored it
      would be guessing beside an answer.
    - `provider_code` — the **upstream's** own error code, where the router surfaced one
      (OpenRouter: ``error.metadata.provider_code``). A 429 on a paid model is the upstream
      limiting, not the router, so this is the field that names who actually refused.
    - `routing_attempt` — how far the router itself got before giving up (OpenRouter:
      ``openrouter_metadata.attempt``), where ``0`` means it reached **no** provider at all. A
      genuine value, so it is never falsified by a truthiness test.
    - `routing_attempts` — the per-upstream result, already rendered compactly as
      ``parasail:429,coreweave:429``. Rendered at the adapter because the shape is one vendor's, and
      because nothing downstream computes on it: it is a breadcrumb naming who was tried and what
      each said, which is the one thing an excluded-from-uptime-stats 429 leaves nowhere else.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        body: str = "",
        retry_after: float | None = None,
        provider_code: str | None = None,
        routing_attempt: int | None = None,
        routing_attempts: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body
        self.retry_after = retry_after
        self.provider_code = provider_code
        self.routing_attempt = routing_attempt
        self.routing_attempts = routing_attempts


class ProviderServerError(ProviderAPIError):
    """The provider failed on *its own side* (HTTP 5xx) — transient, and therefore retried.

    One of three members of the retryable class, alongside `ProviderResponseError` and — since issue
    #506 — `ProviderRateLimitError`. A 5xx is
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

    **It is conceptually a `ProviderRequestError`** (below) — the identical request fails
    identically forever — but it is deliberately *not* a subclass of it, because it has a **better**
    remedy than the generic "report and give up": it self-heals. A wake reports it to the timeline
    only in the residual case where that self-heal could not run (no safe compaction cut), so the
    failure taxonomy treats it as the permanent class's sibling, not a member (issue #336).
    """


class ProviderRequestError(ProviderAPIError):
    """Permanent-for-the-request: the identical request fails identically forever (issue #336).

    The category of provider fault where re-issuing the same request only repeats the rejection —
    between the *transient* faults (retried) and the *account-blocked* one (`ProviderBillingError`,
    which heals when a human funds the account). The wake reports such a fault once to the timeline —
    the *verbatim* vendor error, never softened (decision 3) — marks the driving item handled so it is
    never re-driven, and exits clean.

    **Its only shipped member is `ProviderPayloadTooLargeError`**, and that narrowing is deliberate.
    A *generic* malformed-request 4xx (a bad ``model_params.json`` key, a serialization bug) is almost
    never a permanent property of the peer's *content* — it is a fixable harness/config defect — so
    marking the peer's message handled would lose it the moment the config is fixed. Those stay a plain
    `ProviderAPIError` and **propagate** (CLAUDE.md → Provider Capabilities: "a bad model_params.json
    key propagates"), leaving the message re-drivable. Only a fault that is genuinely permanent for the
    *content itself* — a payload too large to ever accept — is a member and reported-and-handled.
    """


class ProviderPayloadTooLargeError(ProviderRequestError):
    """The request body was too large for the provider to accept (issue #336).

    The 2026-07-21 @briggs incident, made a first-class failure: the founder uploaded a ~19 MB
    photo, base64-inflated past the ``xai-sdk``'s hardcoded 20 MiB gRPC send cap
    (``CLIENT: Sent message larger than max``). It is a `ProviderRequestError` — the identical bytes
    are rejected identically forever — surfaced as its own type so the timeline report can name the
    cause plainly and suggest a smaller or cropped version.

    **The original file is never modified anywhere** (decision 1, Active Storage precedent): the
    harness attempts the original bytes honestly, relays the vendor's verdict, and leaves the human
    to decide whether to reduce or crop. No downscaling, no recompression. Signalled as HTTP 413 by
    OpenAI/OpenRouter-shaped endpoints; as a client-side ``RESOURCE_EXHAUSTED`` gRPC error by the
    native xAI SDK (which computes the message-size overflow before the wire).
    """


class ProviderBillingError(ProviderAPIError):
    """Account-blocked: the provider refused the request for lack of credit / quota (issue #336).

    The third class of the taxonomy, and a **sibling of the rate-limit class, never a variant of
    it** — the distinction is the whole point. A rate limit heals with *time* (the same request
    succeeds a moment later); this heals only with a *human action* (funding the account), and until
    then the same request fails the same way. So a wake does not retry it and does not silently loop:
    it reports the outage to the timeline in plain language a non-technical human understands — "add
    money to this agent's vendor account" — and a peer AI on the timeline can read it and stop
    (decision 5). The report is **debounced** (one notice per outage, then fail fast and quiet) and
    the pending work is left pending, so it resumes untouched the moment the account is funded and a
    call succeeds again (`basecradle_harness._report.BillingState`, `_wake`).

    Vendor signals, each mapped by its own adapter: OpenAI — **HTTP 429 with
    ``error.type == "insufficient_quota"``** (the one 429 that is *not* a rate limit); OpenRouter —
    **HTTP 402 Payment Required**; xAI (native gRPC) — a ``RESOURCE_EXHAUSTED`` whose detail names
    credit/quota (the exact wire shape is unconfirmed from xAI's docs, so it is matched defensively —
    see `basecradle_harness._faults.is_out_of_funds`).
    """


class ProviderRateLimitError(ProviderAPIError):
    """The provider rate-limited the request (HTTP 429) — transient, and therefore retried.

    The third member of the retryable class, and the newest (issue #506). It was deliberately
    *excluded* until 2026-09-16 on the reasoning that hammering a rate-limited endpoint only deepens
    the hole — sound for an unbounded retry against a single server, and wrong for what the fleet
    actually runs: OpenRouter **re-routes** on the re-issued request, so a 429 from one pinned
    upstream is not a statement about the pool. The live evidence settled it — four reranks fell
    back to plain hybrid on 429s that cleared inside ~1.3 seconds, one of them while a sibling
    agent's rerank succeeded on another endpoint *in the same second*.

    What keeps the old reasoning honest is the bound: at most two retries and at most three seconds
    of total sleep (`basecradle_harness._retry`), with the vendor's own ``Retry-After`` preferred
    over the schedule and clamped to what is left. This is a wait, not a hammer.

    `retry_after`, and the routing diagnostics beside it, are declared on `ProviderAPIError` — see
    there for why they belong to every status error rather than to this one.
    """
