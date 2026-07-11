"""The shared log seam: the key=value formatter, token normalization, and the LLM/media lines.

These pin the *contract the journal reads* — Better Stack's Live Tail greps these lines, so
their shape is an interface, not an implementation detail. Nothing here touches a model or the
platform: `token_counts` is fed the usage shapes each vendor SDK actually returns, and the two
emitters are asserted through `caplog`.
"""

import logging

from basecradle_harness._observability import (
    delivery_id,
    describe_provider,
    kv,
    log_llm_call,
    log_media_call,
    media_timer,
    token_counts,
)

# --- the key=value formatter -------------------------------------------------


def test_kv_writes_pairs_in_order_and_drops_the_empty_ones():
    assert kv(timeline="abc", trigger=None, provider="xai", delivery="") == (
        "timeline=abc provider=xai"
    )


def test_kv_keeps_a_zero_because_posting_nothing_is_a_fact():
    assert kv(posted=0) == "posted=0"


# --- token normalization across the vendors ----------------------------------


def test_the_responses_usage_shape_is_read():
    usage = {"input_tokens": 1200, "output_tokens": 64, "total_tokens": 1264}

    assert token_counts(usage) == {"tokens_in": 1200, "tokens_out": 64, "tokens_total": 1264}


def test_the_chat_usage_shape_is_read():
    usage = {"prompt_tokens": 90, "completion_tokens": 10, "total_tokens": 100}

    assert token_counts(usage) == {"tokens_in": 90, "tokens_out": 10, "tokens_total": 100}


def test_a_proto_usage_object_is_read_by_attribute():
    class Usage:  # the xAI gRPC shape: attributes, not keys
        prompt_tokens = 7
        completion_tokens = 3
        total_tokens = 10

    assert token_counts(Usage()) == {"tokens_in": 7, "tokens_out": 3, "tokens_total": 10}


def test_a_provider_that_reports_no_usage_logs_no_token_fields():
    assert token_counts(None) == {}
    assert token_counts({}) == {}


def test_a_non_integer_token_field_is_left_out_rather_than_guessed_at():
    assert token_counts({"input_tokens": "lots", "output_tokens": 5}) == {"tokens_out": 5}


# --- the LLM line ------------------------------------------------------------


def test_the_llm_line_names_the_provider_model_duration_and_tokens(caplog):
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        log_llm_call(
            provider="xai",
            model="grok-4.3",
            seconds=2.5,
            usage={"prompt_tokens": 90, "completion_tokens": 10, "total_tokens": 100},
        )

    assert caplog.records[0].getMessage() == (
        "llm provider=xai model=grok-4.3 duration=2.50s tokens_in=90 tokens_out=10 tokens_total=100"
    )


def test_the_llm_line_still_lands_when_the_sdk_reports_no_usage(caplog):
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        log_llm_call(provider="openai", model="gpt-5.4-mini", seconds=0.4)

    assert caplog.records[0].getMessage() == (
        "llm provider=openai model=gpt-5.4-mini duration=0.40s"
    )


# --- the media line ----------------------------------------------------------


def test_the_media_line_names_the_provider_kind_model_and_duration(caplog):
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        log_media_call(
            provider="xai", kind="video.generate", model="grok-imagine-video", seconds=61
        )

    assert caplog.records[0].getMessage() == (
        "media provider=xai kind=video.generate model=grok-imagine-video duration=61.00s"
    )


def test_the_media_timer_logs_a_completed_generation(caplog):
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        with media_timer(provider="openai", kind="image.generate", model="gpt-image-2"):
            pass

    line = caplog.records[0].getMessage()
    assert line.startswith("media provider=openai kind=image.generate model=gpt-image-2 duration=")


def test_a_failed_generation_logs_nothing_and_lets_the_error_through(caplog):
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        try:
            with media_timer(provider="openai", kind="image.generate", model="gpt-image-2"):
                raise RuntimeError("the API said no")
        except RuntimeError:
            pass

    # The tool relays the failure to the model and the engine's tool line records it; a media
    # line here would time a call that never produced anything.
    assert not caplog.records


# --- the delivery id and the provider descriptor -----------------------------


def test_the_delivery_id_is_read_from_the_routers_env_var(monkeypatch):
    monkeypatch.setenv("BASECRADLE_DELIVERY_ID", "01996f0e-3d2b-7a41-9c5f-2e6a7b8c9d0e")

    assert delivery_id() == "01996f0e-3d2b-7a41-9c5f-2e6a7b8c9d0e"


def test_an_absent_or_blank_delivery_id_is_simply_none(monkeypatch):
    monkeypatch.delenv("BASECRADLE_DELIVERY_ID", raising=False)
    assert delivery_id() is None

    monkeypatch.setenv("BASECRADLE_DELIVERY_ID", "   ")
    assert delivery_id() is None


def test_a_provider_is_described_by_its_vendor_and_model():
    class Adapter:
        provider = "openrouter"
        model = "z-ai/glm-5.2"

    assert describe_provider(Adapter()) == ("openrouter", "z-ai/glm-5.2")


def test_a_third_party_adapter_without_the_labels_still_runs():
    class BareAdapter:  # satisfies `Provider` (a `chat` method) and nothing more
        def chat(self, messages, tools=None): ...

    # Observability never breaks a turn: an unlabeled adapter logs `unknown`, not a crash.
    assert describe_provider(BareAdapter()) == ("unknown", "unknown")
