"""The mechanical failure reporter's building blocks — classify, verbatim, body, debounce (issue #336).

These are the model-free pieces the wake assembles into a timeline post when a provider fails
permanently or for lack of funds. The delivery/settle wiring is exercised in `test_wake.py`; this
pins the parts in isolation: which class an exception falls in, that the vendor error is relayed
*verbatim*, the two peer-facing shapes, and the per-timeline debounce + self-heal.
"""

from __future__ import annotations

from basecradle_harness._exceptions import (
    ProviderAPIError,
    ProviderAuthError,
    ProviderBillingError,
    ProviderContextLengthError,
    ProviderError,
    ProviderPayloadTooLargeError,
    ProviderRateLimitError,
    ProviderResponseError,
)
from basecradle_harness._report import (
    BILLING,
    PERMANENT,
    BillingState,
    classify,
    provider_label,
    report_body,
    verbatim,
)


def test_classify_maps_each_reported_class_and_only_those():
    assert classify(ProviderBillingError("x", status_code=402)).kind == BILLING
    assert classify(ProviderPayloadTooLargeError("x", status_code=413)).kind == PERMANENT
    assert classify(ProviderContextLengthError("x", status_code=400)).kind == PERMANENT
    assert (
        classify(ProviderPayloadTooLargeError("x", status_code=413)).reason == "payload_too_large"
    )
    assert classify(ProviderContextLengthError("x", status_code=400)).reason == "context_length"
    assert classify(ProviderBillingError("x", status_code=402)).reason == "out_of_funds"
    # Not reported: a transient fault, an auth error, a rate limit, and a **generic** malformed-request
    # `ProviderAPIError` (which propagates, a fixable config defect — not permanent-for-content). The
    # wake leaves all of these on their existing (non-reporting) paths.
    assert classify(ProviderAPIError("x", status_code=400)) is None
    assert classify(ProviderRateLimitError("x", status_code=429)) is None
    assert classify(ProviderAuthError("x", status_code=401)) is None
    assert classify(ProviderResponseError("x")) is None
    assert classify(ValueError("nope")) is None


def test_verbatim_digs_the_vendor_message_out_of_a_json_body():
    exc = ProviderPayloadTooLargeError(
        "Provider returned HTTP 413",
        status_code=413,
        body='{"error": {"message": "Request too large: 25 MB exceeds the 20 MB limit"}}',
    )
    assert verbatim(exc) == "Request too large: 25 MB exceeds the 20 MB limit"


def test_verbatim_falls_back_to_str_for_a_gRPC_shaped_error():
    # The native xAI path has no JSON body — its own str IS the verbatim vendor line.
    exc = ProviderPayloadTooLargeError(
        "xAI gRPC error (RESOURCE_EXHAUSTED): CLIENT: Sent message larger than max (25470493 vs. 20971520)",
        status_code=413,
    )
    assert "Sent message larger than max (25470493 vs. 20971520)" in verbatim(exc)


def test_provider_label_reads_known_vendors_and_passes_unknown_through():
    assert provider_label("xai") == "xAI"
    assert provider_label("openai") == "OpenAI"
    assert provider_label("openrouter") == "OpenRouter"
    assert provider_label("acme") == "acme"
    assert provider_label(None) == "the model provider"


def test_report_body_permanent_names_the_item_and_relays_verbatim():
    exc = ProviderPayloadTooLargeError(
        "too big",
        status_code=413,
        body='{"error": {"message": "Sent message larger than max (25 MB vs. 20 MB)"}}',
    )
    rc = classify(exc)
    body = report_body(rc, item="the file you shared (cat.jpg)", provider="xai", exc=exc)
    assert "the file you shared (cat.jpg)" in body
    assert "xAI" in body
    assert "Sent message larger than max (25 MB vs. 20 MB)" in body  # verbatim, unsoftened
    assert (
        "untouched" in body
    )  # decision 1: the original is not modified; a smaller version may work


def test_report_body_billing_says_add_funds_in_plain_language():
    exc = ProviderBillingError(
        "no money",
        status_code=402,
        body='{"error": {"message": "Insufficient credits. Add more at openrouter.ai/credits"}}',
    )
    rc = classify(exc)
    body = report_body(rc, item="your message", provider="openrouter", exc=exc)
    assert "out of credit" in body
    assert "Add funds" in body
    assert "OpenRouter" in body
    assert "Insufficient credits. Add more at openrouter.ai/credits" in body  # verbatim


def test_billing_state_debounces_one_notice_per_outage_per_timeline(tmp_path):
    state = BillingState(tmp_path)
    tl = "0198b1f0-0000-7000-8000-000000000001"
    # First failure of the outage → notify. Every one after → stay quiet.
    assert state.note_and_check(tl) is True
    assert state.note_and_check(tl) is False
    assert state.note_and_check(tl) is False
    assert state.blocked(tl) is True


def test_billing_state_is_per_timeline(tmp_path):
    state = BillingState(tmp_path)
    a = "0198b1f0-0000-7000-8000-00000000000a"
    b = "0198b1f0-0000-7000-8000-00000000000b"
    assert state.note_and_check(a) is True
    # A different timeline is its own outage — it still deserves its one notice.
    assert state.note_and_check(b) is True
    assert state.note_and_check(a) is False


def test_billing_state_self_heals_and_re_notifies_the_next_outage(tmp_path):
    state = BillingState(tmp_path)
    tl = "0198b1f0-0000-7000-8000-000000000001"
    state.note_and_check(tl)
    # A call got through → clear the marker (returns True: a recovery happened).
    assert state.recovered(tl) is True
    assert state.blocked(tl) is False
    # A healthy wake clears nothing and does not claim a recovery.
    assert state.recovered(tl) is False
    # The next outage notifies from a clean slate.
    assert state.note_and_check(tl) is True


def test_report_body_context_length_names_the_window():
    exc = ProviderContextLengthError("over", status_code=400, body="context length exceeded")
    body = report_body(classify(exc), item="your message", provider="openai", exc=exc)
    assert "too long for my context window" in body


def test_verbatim_of_a_plain_provider_error_is_its_str():
    assert verbatim(ProviderError("plain cause")) == "plain cause"
