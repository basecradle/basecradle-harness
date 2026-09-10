"""The provider-failure text heuristics — out-of-funds and payload-too-large (issue #336).

Both are the sibling of `is_context_overflow`: a phrase match that classifies a provider error by the
nature of the fault, and both **fail safe** — an unrecognized phrasing returns ``False`` so the
adapter keeps its existing classification. The patterns are kept narrow, so these pin both the
positives (real vendor wordings) and the negatives (a rate limit must not read as an outage).
"""

from __future__ import annotations

import pytest

from basecradle_harness._faults import is_out_of_funds, is_too_large, refused_tool_schema


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


# --- refused_tool_schema: which tool the vendor refused (issue #496) ----------
#
# xAI states this in two shapes, and the second one is why the matcher is not one literal. The issue
# report quoted a bracketed error code; the live endpoint, asked the same question with the same
# schema on the same day, sent no code at all. A matcher anchored only on the code would have
# returned None on every real refusal — the reactive drop dead, and reporting nothing.

_LIVE_DETAIL = (
    "workmail__send_email: tool parameter root must be an object type "
    "(root schema is an anyOf/oneOf union with a non-object branch)"
)
_REPORTED_DETAIL = "Failed to start sampling: [invalid_client_tool_schema] " + _LIVE_DETAIL


@pytest.mark.parametrize("text", [_LIVE_DETAIL, _REPORTED_DETAIL])
def test_both_shapes_xai_states_it_in_name_the_tool(text):
    found = refused_tool_schema(text)
    assert found is not None
    name, reason = found
    assert name == "workmail__send_email"
    assert reason.startswith("tool parameter root must be an object type")


@pytest.mark.parametrize(
    "text",
    [
        "",
        "prompt is too long: 300000 tokens > 256000 maximum context length",
        "model: not found",  # `NAME: reason`-shaped, but nothing says it is about a tool schema
        "Rate limit exceeded",
        "boom",
    ],
)
def test_an_error_that_is_not_a_tool_schema_refusal_is_not_read_as_one(text):
    assert refused_tool_schema(text) is None


def test_the_bare_form_needs_the_name_where_the_vendor_puts_it():
    # A tool-schema signal is necessary but not sufficient: without a leading `NAME:` there is no
    # tool to drop, and guessing one would drop the wrong capability.
    assert refused_tool_schema("tool parameter root must be an object type") is None
