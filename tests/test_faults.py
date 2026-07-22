"""The provider-failure text heuristics — out-of-funds and payload-too-large (issue #336).

Both are the sibling of `is_context_overflow`: a phrase match that classifies a provider error by the
nature of the fault, and both **fail safe** — an unrecognized phrasing returns ``False`` so the
adapter keeps its existing classification. The patterns are kept narrow, so these pin both the
positives (real vendor wordings) and the negatives (a rate limit must not read as an outage).
"""

from __future__ import annotations

import pytest

from basecradle_harness._faults import is_out_of_funds, is_too_large


@pytest.mark.parametrize(
    "text",
    [
        "You exceeded your current quota, please check your plan and billing details.",
        "insufficient_quota",
        "Insufficient credit to complete this request",
        "Your account is out of credits",
        "no remaining balance on this account",
        "Please add credit to continue",
        "402 Payment Required",
        "prepaid credit balance is $0.00",
    ],
)
def test_is_out_of_funds_recognizes_billing_wordings(text):
    assert is_out_of_funds(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Rate limit exceeded",
        "429 Too Many Requests, retry after 30s",
        "too many requests per minute",
        "The model is overloaded, try again",
        "context length exceeded",  # a different fault (compaction's job), never billing
        "Sent message larger than max (25470493 vs. 20971520)",  # too-large, not billing
    ],
)
def test_is_out_of_funds_stays_quiet_on_non_billing(text):
    assert is_out_of_funds(text) is False


@pytest.mark.parametrize(
    "text",
    [
        "CLIENT: Sent message larger than max (25470493 vs. 20971520)",  # the @briggs incident
        "message larger than max",
        "Request entity too large",
        "payload too large",
        "image too large",
        "exceeds the maximum allowed request size",
    ],
)
def test_is_too_large_recognizes_payload_wordings(text):
    assert is_too_large(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Rate limit exceeded",
        "context length exceeded",  # a context overflow routes to compaction, not a report
        "too many input tokens",
        "insufficient_quota",  # billing, not too-large
    ],
)
def test_is_too_large_stays_quiet_on_other_faults(text):
    assert is_too_large(text) is False
