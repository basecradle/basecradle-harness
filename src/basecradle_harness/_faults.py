"""Recognize the *nature* of a provider failure from its error text — the taxonomy's heuristics.

The three-way provider-failure taxonomy (issue #336) classifies every fault by its nature, never by
the vendor: transient (retried), permanent-for-the-request (reported once, never retried), and
account-blocked / out-of-funds (reported, debounced, self-healing). Most of the wire signals an
adapter maps are **structured** — an HTTP status, an ``error.type`` code — and those are read
directly where they live. Two signals are not, and this module holds their heuristics:

- `is_out_of_funds` — the billing / insufficient-credit class, for the one adapter whose out-of-funds
  shape is *not* a clean status code: the native xAI gRPC path overloads ``RESOURCE_EXHAUSTED`` for
  rate limits, client-side message-too-large, *and* (plausibly) credit exhaustion, so it must be
  disambiguated on the detail string. xAI's exact out-of-credit gRPC shape could not be confirmed
  from its published docs, so this is matched **defensively** (issue #336 says to, and to say so).
- `is_too_large` — the payload-too-large class, for the same xAI gRPC path, where the ``xai-sdk``'s
  own 20 MiB channel cap raises a *client-side* ``RESOURCE_EXHAUSTED`` reading ``Sent message larger
  than max`` before anything reaches the wire (the 2026-07-21 @briggs incident).

Both are the sibling of `basecradle_harness._context.is_context_overflow` — a phrase match on a
provider error string — and both **fail safe** exactly as it does: a phrasing they do not recognize
is simply not recognized, and the adapter falls through to its existing classification (a bare
``RESOURCE_EXHAUSTED`` stays a rate limit). A false negative costs one misclassified fault; the
patterns are kept narrow so a false positive — reporting a rate limit as an out-of-funds outage — is
the rarer mistake.
"""

from __future__ import annotations

import re

#: Phrases that mean *the account cannot pay for this request* — out of credit / quota / funds.
#: Anchored on the money words a billing rejection uses, kept clear of the ones a *rate* limit uses
#: ("rate", "per minute", "too many requests"), so a genuine 429 is never read as an outage. "quota"
#: is included because OpenAI's own out-of-funds body says ``insufficient_quota`` — but the adapter
#: reads that structured code first and only falls back to this text match for a vendor (xAI) that
#: gives no code. Matches OpenAI's human message ("exceeded your current quota, please check your
#: plan and billing details") and the plain shapes an xAI/OpenRouter body is likely to use.
_OUT_OF_FUNDS_PHRASES = re.compile(
    r"insufficient[ _-]?(?:quota|funds|credit|balance)"
    r"|out of (?:credit|credits|funds|quota)"
    r"|no (?:remaining )?(?:credit|credits|funds|balance)"
    r"|(?:add|top[ -]?up|purchase|buy) (?:more )?credit"
    r"|(?:credit|prepaid) balance"
    r"|payment required"
    r"|billing (?:details|hard limit|issue)"
    r"|check your plan and billing"
    r"|exceeded your current quota",
    re.IGNORECASE,
)

#: Phrases that mean *the request body was too large to send* — the payload-too-large class. Written
#: for the client-side gRPC shape the ``xai-sdk`` raises (``Sent message larger than max (X vs. Y)``),
#: plus the plain HTTP wordings a 413-shaped body uses. Kept off "context"/"token" wordings, which are
#: `is_context_overflow`'s job and route to the self-healing compaction path instead.
_TOO_LARGE_PHRASES = re.compile(
    r"sent message larger than max"
    r"|message larger than max"
    r"|(?:request|payload|body|message|file|image) (?:entity )?too large"
    r"|exceeds the maximum (?:allowed )?(?:request |payload |upload )?size"
    r"|payload size exceeds",
    re.IGNORECASE,
)


def is_out_of_funds(text: str) -> bool:
    """Does this provider error text say the account is out of credit / quota (issue #336)?

    Fails safe: an unrecognized phrasing returns ``False`` and the adapter keeps its existing
    classification (a bare gRPC ``RESOURCE_EXHAUSTED`` stays a rate limit). See `_OUT_OF_FUNDS_PHRASES`
    for why the pattern is deliberately narrow.
    """
    return bool(text) and bool(_OUT_OF_FUNDS_PHRASES.search(text))


def is_too_large(text: str) -> bool:
    """Does this provider error text say the request body was too large to send (issue #336)?

    The client-side ``xai-sdk`` message-size overflow and the HTTP-413 wordings. Fails safe (an
    unrecognized phrasing returns ``False``), and stays clear of context/token wordings so a genuine
    context overflow still routes to compaction rather than an untouched-file report.
    """
    return bool(text) and bool(_TOO_LARGE_PHRASES.search(text))
