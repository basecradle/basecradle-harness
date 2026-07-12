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
    reported_cost,
    serving_endpoint,
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


# --- the cached-prompt count (issue #274) ------------------------------------
#
# Whether prompt caching is doing anything at all was, until this field, only inferable — and the
# inference was wrong: OpenRouter's own `supports_implicit_caching` reads false on endpoints where
# caching demonstrably works. The reported count is the only trustworthy witness.


def test_the_chat_wire_reports_cached_tokens_under_a_details_block():
    usage = {
        "prompt_tokens": 300020,
        "completion_tokens": 236,
        "total_tokens": 300256,
        "prompt_tokens_details": {"cached_tokens": 238277},
    }

    assert token_counts(usage)["cached_tokens"] == 238277


def test_the_responses_wire_reports_it_under_its_own_details_block():
    usage = {
        "input_tokens": 1200,
        "output_tokens": 64,
        "input_tokens_details": {"cached_tokens": 1024},
    }

    assert token_counts(usage)["cached_tokens"] == 1024


def test_the_xai_proto_reports_it_flat_under_its_own_name():
    class Usage:  # the xAI gRPC shape: attributes, and its own spelling
        prompt_tokens = 7
        completion_tokens = 3
        total_tokens = 10
        cached_prompt_text_tokens = 4

    assert token_counts(Usage())["cached_tokens"] == 4


def test_a_provider_that_reports_no_cached_count_logs_no_cached_field():
    assert "cached_tokens" not in token_counts({"prompt_tokens": 90, "completion_tokens": 10})
    # A details block the vendor sent but left empty is the same story: nothing to report.
    assert "cached_tokens" not in token_counts({"prompt_tokens": 90, "prompt_tokens_details": None})


# --- the cost, where the provider states one ---------------------------------


def test_the_cost_is_read_where_the_provider_reports_dollars():
    assert reported_cost({"prompt_tokens": 1, "cost": 0.0445}) == 0.0445


def test_a_provider_that_reports_tokens_but_no_dollars_gets_no_cost():
    """The harness ships no price table — a stale one is worse than an absent field, and the
    dollar math for a token-only provider belongs at the dashboard layer."""
    assert reported_cost({"prompt_tokens": 90, "completion_tokens": 10}) is None
    assert reported_cost(None) is None


def test_a_non_numeric_cost_is_left_out_rather_than_guessed_at():
    assert reported_cost({"cost": "cheap"}) is None
    assert reported_cost({"cost": True}) is None  # a bool is not a dollar figure


# --- the serving endpoint: a capability, not a vendor branch ------------------


def _routed(selected: str, *considered: str, provider: str | None = None) -> dict:
    """A response carrying OpenRouter's routing metadata — the endpoints it weighed, and its pick.

    `provider` sets the response's **top-level** `provider` field, which is the one that must never
    be read (issue #280); the tests pass a *wrong* value there on purpose.
    """
    available = [{"provider": name, "selected": False} for name in considered]
    available.append({"provider": selected, "selected": True})
    body = {
        "id": "gen-1",
        "model": "z-ai/glm-5.2",
        "openrouter_metadata": {"endpoints": {"available": available, "total": len(available)}},
    }
    if provider is not None:
        body["provider"] = provider
    return body


def test_the_serving_endpoint_is_the_one_the_router_says_it_selected():
    """OpenRouter is a *router*: one model id fans out to endpoints differing 10× in context ceiling
    and 5.4× in price, so `provider=openrouter` alone cannot say what a call ran against. The answer
    is the endpoint the router flags `selected` in its routing metadata."""
    assert serving_endpoint(_routed("StreamLake", "Novita", "Together")) == "StreamLake"


def test_the_top_level_provider_field_is_never_read():
    """The defect of issue #280, pinned so it cannot come back.

    The response's top-level `provider` is undocumented and does **not** mean the serving endpoint:
    it names the last upstream OpenRouter spoke to, which — whenever a server-side tool ran — is the
    tool's provider, not the model's. Live, with `openrouter:web_search` active, a `z-ai/glm-5.2`
    call really does return `"provider": "OpenAI"` while the routing metadata correctly reports
    `StreamLake`. Reading it fabricated a routing distribution that never happened.
    """
    body = _routed("StreamLake", "Novita", provider="OpenAI")

    assert serving_endpoint(body) == "StreamLake"


def test_an_unselected_pool_names_no_endpoint_rather_than_guessing():
    """The router listed what it *could* have used but flagged none as used — so we know the pool,
    not the pick. Omit the field; never fall back to the top-level `provider` that produced #280."""
    body = {
        "provider": "OpenAI",
        "openrouter_metadata": {"endpoints": {"available": [{"provider": "Novita"}]}},
    }

    assert serving_endpoint(body) is None


def test_a_direct_to_vendor_response_names_no_endpoint_and_the_field_is_omitted():
    """Where the SDK talks straight to the vendor, the vendor *is* the endpoint — the honest
    answer is nothing, not a restatement of the provider."""
    assert serving_endpoint({"id": "resp_1", "model": "gpt-5.4-mini"}) is None
    assert serving_endpoint(None) is None

    class Proto:  # the xAI gRPC response: no such concept
        content = "Hi."

    assert serving_endpoint(Proto()) is None


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


def test_the_llm_line_names_the_serving_endpoint_the_cache_hit_and_the_cost(caplog):
    """The full line a router's call earns (issue #274): who dispatched it, who *served* it, what
    it cost, and how much of the prompt was a cache hit rather than full freight."""
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        log_llm_call(
            provider="openrouter",
            model="z-ai/glm-5.2",
            seconds=42.96,
            usage={
                "prompt_tokens": 764942,
                "completion_tokens": 236,
                "total_tokens": 765178,
                "prompt_tokens_details": {"cached_tokens": 238277},
                "cost": 0.0445,
            },
            endpoint="StreamLake",
            cost=0.0445,
        )

    assert caplog.records[0].getMessage() == (
        "llm provider=openrouter endpoint=StreamLake model=z-ai/glm-5.2 duration=42.96s "
        "tokens_in=764942 tokens_out=236 tokens_total=765178 cached_tokens=238277 cost=0.0445"
    )


def test_a_free_call_reports_a_zero_cost_rather_than_dropping_the_field(caplog):
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        log_llm_call(provider="openrouter", model="free/model", seconds=1, cost=0.0)

    assert "cost=0" in caplog.records[0].getMessage()


def test_a_sub_cent_cost_is_plain_fixed_point_not_scientific_notation(caplog):
    """`str(4.4e-05)` is what a grep of this stream cannot read as money."""
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        log_llm_call(provider="xai", model="grok-4.3", seconds=1, cost=0.000044)

    assert "cost=0.000044" in caplog.records[0].getMessage()


def test_a_cost_that_is_not_a_number_is_dropped_rather_than_crashing_the_wake(caplog):
    """The figure comes straight off a vendor object, so a surprise there must cost the *field*,
    never the turn — the log seam is the last place that should raise."""
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        log_llm_call(provider="xai", model="grok-4.3", seconds=1, cost="free")  # type: ignore[arg-type]

    assert "cost=" not in caplog.records[0].getMessage()


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


# --- foreign text is rendered, never interpolated (issue #272 review) ---------


def test_a_newline_in_an_error_cannot_split_the_record():
    """A multi-line exception (an MCP server relaying stderr, a subprocess error) would otherwise
    become several journald lines — and the continuation lines carry no level, so a WARNING filter
    would show a decapitated fragment."""
    line = kv(name="shell", error="boom\nTraceback (most recent call last):\n  File ...")

    assert "\n" not in line
    assert line.startswith("name=shell error=")


def test_an_error_cannot_forge_a_field():
    """A tool's exception text is not the harness's text. Unquoted, a message containing
    `outcome=ok` would parse as a field the harness never wrote."""
    line = kv(name="evil", error="failed outcome=ok duration=0.00s")

    # The whole message is one quoted value — a key=value parser sees `error`, not `outcome`.
    assert line == 'name=evil error="failed outcome=ok duration=0.00s"'


def test_a_credential_shape_is_scrubbed_before_it_reaches_the_journal():
    """Defense at the source: an exception from a drop-in tool can embed the request URL or an
    Authorization header, and a provider's 4xx can echo the key it rejected."""
    line = kv(
        error="401 from https://api.x.ai/v1?api_key=xai-abcd1234efgh5678 (Bearer sk-live-abcd1234efgh)"
    )

    assert "xai-abcd1234efgh5678" not in line
    assert "sk-live-abcd1234efgh" not in line
    assert line.count("[redacted]") == 2


def test_the_platform_token_shape_is_scrubbed_too():
    assert "bc_uat_KqI8zFxkQ0OZ8vYwT7mWcVtR3nSdLpEa" not in kv(
        error="rejected token bc_uat_KqI8zFxkQ0OZ8vYwT7mWcVtR3nSdLpEa"
    )


def test_a_long_value_is_bounded():
    """A log line is a breadcrumb to the failure, not a copy of it — the model already got the
    full error as its tool result."""
    line = kv(error="x" * 5000)

    assert len(line) < 300
    assert line.endswith("…")  # a bare (space-free) value needs no quoting — just the ellipsis

    # A prose error — the realistic case — is bounded *and* quoted, so the truncation can never
    # leave a dangling half-field for a parser to trip on.
    prose = kv(error="the tool exploded " * 100)
    assert len(prose) < 300
    assert prose.endswith('…"')


def test_a_bare_token_value_stays_unquoted_so_the_common_line_greps_cleanly():
    assert kv(timeline="019e7750-66ee-7f53-829f-13a8a710b6da", posted=3, duration="1.20s") == (
        "timeline=019e7750-66ee-7f53-829f-13a8a710b6da posted=3 duration=1.20s"
    )
