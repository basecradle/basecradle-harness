"""The shared log seam: the key=value formatter, token normalization, and the LLM and tool lines.

These pin the *contract the journal reads* — Better Stack's Live Tail greps these lines, so
their shape is an interface, not an implementation detail. Nothing here touches a model or the
platform: `token_counts` is fed the usage shapes each vendor SDK actually returns, and the two
emitters are asserted through `caplog`.
"""

import logging
import re
from datetime import UTC, datetime

import pytest

from basecradle_harness import Tool, ToolRegistry
from basecradle_harness._engine import Engine
from basecradle_harness._observability import (
    BLUE,
    GREEN,
    NO_COLOR_ENV,
    RED,
    RESET,
    YELLOW,
    ComputedCost,
    _full_value,
    _value,
    byok,
    capture_llm_call,
    color_enabled,
    delivery_id,
    describe_provider,
    finish_reason,
    head,
    kv,
    log_cost_unpriced,
    log_llm_call,
    log_server_tool,
    log_tool,
    log_unspoken,
    meter_tool,
    metered,
    reported_cost,
    routed_web_searches,
    serving_endpoint,
    slug,
    tally_server_tools,
    token_counts,
    tool_meter,
    truncated,
    usage_reported,
)
from tests.conftest import plain as unstyled
from tests.conftest import tool_lines

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


# --- bring-your-own-key: the provider bills it, and the dollars move (issue #653) ---------------
#
# On a BYOK call OpenRouter's own `usage.cost` is 0 — its credits were not spent — and the
# provider's charge is `cost_details.upstream_inference_cost`. Measured 2026-10-06 on
# `google/gemini-3.6-flash` through a Vertex BYOK credential; these figures are that response's.

#: The usage block of a BYOK call, as OpenRouter returned it.
BYOK_USAGE = {
    "prompt_tokens": 42,
    "completion_tokens": 111,
    "total_tokens": 153,
    "cost": 0,
    "is_byok": True,
    "cost_details": {
        "upstream_inference_cost": 0.000288,
        "upstream_inference_prompt_cost": 0.0000105,
        "upstream_inference_completions_cost": 0.0002775,
    },
}


def test_a_byok_call_costs_what_the_provider_bills_for_it():
    """`cost=` is what the operator pays, whichever party bills it — so on a BYOK call it is the
    upstream figure, and never the sum with OpenRouter's 0 beside it."""
    assert byok(BYOK_USAGE) is True
    assert reported_cost(BYOK_USAGE) == 0.000288


def test_a_byok_block_that_states_no_upstream_figure_reports_no_cost():
    """The `cost: 0` beside it is OpenRouter's share. Printing it would turn a call the provider
    billed into a free one, so the field is absent — as for any provider that states no dollars."""
    usage = {k: v for k, v in BYOK_USAGE.items() if k != "cost_details"}
    assert reported_cost(usage) is None
    assert reported_cost(usage | {"cost_details": None}) is None
    assert reported_cost(usage | {"cost_details": {"upstream_inference_cost": None}}) is None


def test_a_call_billed_the_ordinary_way_reads_usage_cost_as_before():
    """No `is_byok` is what a non-BYOK call carries; an explicit `false` reads the same — and an
    `upstream_inference_cost` beside it is not the operator's bill, so it is never preferred."""
    ordinary = {"prompt_tokens": 1, "cost": 0.0445}
    assert byok(ordinary) is False
    assert reported_cost(ordinary) == 0.0445
    explicit = ordinary | {"is_byok": False, "cost_details": {"upstream_inference_cost": 0.04}}
    assert byok(explicit) is False
    assert reported_cost(explicit) == 0.0445


def test_only_a_literal_true_reads_as_byok():
    """A vendor's surprise must not move the dollar figure to a field that is not there."""
    for surprise in ("true", 1, "yes"):
        assert byok({"is_byok": surprise}) is False
    assert byok(None) is False

    class Usage:  # an SDK object rather than a mapping reads the same way
        is_byok = True
        cost = 0
        cost_details = type("CostDetails", (), {"upstream_inference_cost": 0.000288})()

    assert byok(Usage()) is True
    assert reported_cost(Usage()) == 0.000288


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

    assert unstyled(caplog.records[0].getMessage()) == (
        "llm provider=xai purpose=main model=grok-4.3 duration=2.50s tokens_in=90 tokens_out=10 "
        "tokens_total=100 outcome=ok"
    )


def test_the_llm_line_still_lands_when_the_sdk_reports_no_usage(caplog):
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        log_llm_call(provider="openai", model="gpt-5.4-mini", seconds=0.4)

    assert unstyled(caplog.records[0].getMessage()) == (
        "llm provider=openai purpose=main model=gpt-5.4-mini duration=0.40s outcome=ok"
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

    assert unstyled(caplog.records[0].getMessage()) == (
        "llm provider=openrouter purpose=main endpoint=StreamLake model=z-ai/glm-5.2 duration=42.96s "
        "tokens_in=764942 tokens_out=236 tokens_total=765178 cached_tokens=238277 cost=0.0445 "
        "outcome=ok"
    )


def test_a_byok_line_carries_the_upstream_cost_and_names_the_biller_last(caplog):
    """`billing=byok` rides after everything, `generation_id` included, so no field the NOC
    already extracts moves on a BYOK line (issue #653)."""
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        log_llm_call(
            provider="openrouter",
            purpose="helper",
            kind="image.describe",
            endpoint="Google",
            model="google/gemini-3.6-flash",
            seconds=1.5,
            usage=BYOK_USAGE,
            cost=reported_cost(BYOK_USAGE),
            extra={"subject": "cat.png"},
            generation_id="gen-1791300000-AbCdEfGhIjKlMnOpQrSt",
        )

    assert unstyled(caplog.records[0].getMessage()) == (
        "llm provider=openrouter purpose=helper kind=image.describe endpoint=Google "
        "model=google/gemini-3.6-flash duration=1.50s tokens_in=42 tokens_out=111 tokens_total=153 "
        "cost=0.000288 outcome=ok subject=cat.png "
        "generation_id=gen-1791300000-AbCdEfGhIjKlMnOpQrSt billing=byok"
    )


def test_a_byok_line_with_no_upstream_figure_still_names_the_biller(caplog):
    """Who bills it is stated; how much is not — so `billing=byok` and no `cost=`."""
    usage = {k: v for k, v in BYOK_USAGE.items() if k != "cost_details"}
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        log_llm_call(
            provider="openrouter",
            model="google/gemini-3.6-flash",
            seconds=1,
            usage=usage,
            cost=reported_cost(usage),
        )

    line = caplog.records[0].getMessage()
    assert "cost=" not in line
    assert line.endswith(" billing=byok")


def test_a_non_byok_line_is_byte_identical_and_carries_no_billing(caplog):
    """The line a call billed the ordinary way carries no ``billing`` at all (issue #653)."""
    usage = {"prompt_tokens": 90, "completion_tokens": 10, "total_tokens": 100, "cost": 0.0445}
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        log_llm_call(
            provider="openrouter",
            model="z-ai/glm-5.2",
            seconds=2.5,
            usage=usage,
            cost=reported_cost(usage),
            generation_id="gen-1",
        )

    assert unstyled(caplog.records[0].getMessage()) == (
        "llm provider=openrouter purpose=main model=z-ai/glm-5.2 duration=2.50s "
        "tokens_in=90 tokens_out=10 tokens_total=100 cost=0.0445 outcome=ok generation_id=gen-1"
    )


def test_billing_is_read_off_usage_and_one_carried_in_extra_is_replaced(caplog):
    """An `extra` naming `billing` must neither raise (two values for one keyword, inside a wake)
    nor speak for the vendor: the field is what usage says, and it stays last."""
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        log_llm_call(
            provider="openrouter",
            model="m",
            seconds=1,
            usage={"prompt_tokens": 1},
            extra={"billing": "byok", "surface": "tool"},
        )
        log_llm_call(
            provider="openrouter", model="m", seconds=1, usage=BYOK_USAGE, extra={"billing": "x"}
        )

    assert caplog.records[0].getMessage().endswith(" surface=tool")
    assert caplog.records[1].getMessage().endswith(" billing=byok")


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


def test_a_negative_or_non_finite_cost_is_omitted_so_the_line_stays_greppable(caplog):
    """`cost=` must stay `cost=([0-9.]+)`-matchable, or the call slips silently out of the
    dashboard's rollup. A negative (`cost=-0.02`) or non-finite (`cost=nan`/`cost=inf`, which
    stdlib `json` decodes without complaint) figure would break that — a charge is never either,
    so the field is omitted rather than logged un-matchable. Both line kinds share `_money`."""
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        log_llm_call(provider="xai", model="grok-4.3", seconds=1, cost=-0.02)
        log_tool(name="grok_generate_video", executor="harness", outcome="ok", cost=-1.0)
        log_llm_call(provider="xai", model="grok-4.3", seconds=1, cost=float("inf"))
        log_tool(name="grok_generate_image", executor="harness", outcome="ok", cost=float("nan"))

    assert all("cost=" not in r.getMessage() for r in caplog.records)


# --- the tool line: one head whoever executed the tool (issue #758) ------------------------------


@pytest.fixture
def plain(monkeypatch):
    """The bytes a line carries with color off, so an exact comparison reads the fields alone."""
    monkeypatch.setenv(NO_COLOR_ENV, "1")


def test_a_harness_tool_line_carries_its_vendor_work_in_the_contracts_order(caplog, plain):
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        log_tool(
            name="grok_generate_video",
            executor="harness",
            outcome="ok",
            seconds=61,
            provider="xai",
            kind="video.generate",
            model="grok-imagine-video-1.5",
            cost=2.1,  # one 15s 720p clip ≈ $2.10 — real dollars, stated by xAI
        )

    assert caplog.records[0].getMessage() == (
        "tool name=grok_generate_video executor=harness provider=xai kind=video.generate "
        "model=grok-imagine-video-1.5 duration=61.00s outcome=ok cost=2.1"
    )


def test_a_platform_tool_line_names_only_what_it_knows(caplog, plain):
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        log_tool(name="messages", executor="harness", outcome="ok", seconds=0.12)

    assert caplog.records[0].getMessage() == (
        "tool name=messages executor=harness duration=0.12s outcome=ok"
    )


def test_a_failed_run_names_its_reason_and_writes_the_error_on_a_warning(caplog, plain):
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        log_tool(
            name="messages",
            executor="harness",
            outcome="error",
            seconds=0.5,
            reason="exception",
            error="the platform said no",
        )

    info, warning = caplog.records
    assert info.levelno == logging.INFO
    assert info.getMessage() == (
        "tool name=messages executor=harness duration=0.50s outcome=error reason=exception"
    )
    assert warning.levelno == logging.WARNING
    assert warning.getMessage() == 'tool name=messages error="the platform said no"'


def test_a_server_tool_line_has_a_count_the_calls_id_and_no_duration(caplog, plain):
    """The work ran inside a model call its `llm` line already timed; the id joins the two."""
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        log_server_tool(
            name="web_search",
            provider="anthropic",
            model="claude-haiku-5-5",
            count=2,
            generation_id="msg_01ContractSearch",
            cost=ComputedCost(0.02),
        )

    (record,) = caplog.records
    assert record.levelno == logging.INFO
    assert record.getMessage() == (
        "tool name=web_search executor=server provider=anthropic kind=search.web "
        "model=claude-haiku-5-5 count=2 outcome=ok cost=0.02 cost_basis=computed "
        "generation_id=msg_01ContractSearch"
    )


def test_a_server_tool_the_vendor_reports_failed_is_a_warning_with_its_code(caplog, plain):
    """No error text to carry, so no companion: the line itself is the WARNING."""
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        log_server_tool(
            name="web_fetch",
            provider="anthropic",
            model="claude-haiku-5-5",
            count=1,
            generation_id="msg_01ContractFetch",
            reason="max_uses_exceeded",
        )

    (record,) = caplog.records
    assert record.levelno == logging.WARNING
    assert record.getMessage() == (
        "tool name=web_fetch executor=server provider=anthropic model=claude-haiku-5-5 count=1 "
        "outcome=error reason=max_uses_exceeded generation_id=msg_01ContractFetch"
    )


def test_server_tool_uses_are_tallied_per_tool_with_the_first_failure_as_a_slug():
    found = tally_server_tools(
        [("code_execution", ""), ("url_context", "URL_RETRIEVAL_STATUS_ERROR"), ("url_context", "")]
    )

    assert {name: (t.count, t.reason) for name, t in found.items()} == {
        "code_execution": (1, None),
        "url_context": (2, "url_retrieval_status_error"),
    }
    assert slug("max_uses_exceeded") == "max_uses_exceeded"
    assert slug("OUTCOME_FAILED") == "outcome_failed"
    assert slug("  ") is None


def test_a_routers_web_search_count_is_read_and_nothing_else_is():
    assert routed_web_searches({"server_tool_use_details": {"web_search_requests": 2}}) == 2
    assert routed_web_searches({"server_tool_use_details": {"web_search_requests": 0}}) == 0
    assert routed_web_searches({"server_tool_use_details": {"web_search_requests": True}}) == 0
    assert routed_web_searches({}) == 0
    assert routed_web_searches(None) == 0


def test_an_unpriced_charge_is_one_warning_naming_the_charge_and_why(caplog):
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        log_cost_unpriced(provider="openai", charge="container_time", reason="not_reported")

    (record,) = caplog.records
    assert record.levelno == logging.WARNING
    assert record.getMessage() == (
        "cost unpriced provider=openai charge=container_time reason=not_reported"
    )


# --- the tool meter: a harness tool's vendor work, onto its run's one line ------------------------
#
# xAI reports the exact charge for image and video generation on the wire, and the OpenAI and
# Google tables compute theirs — real money the tool-cost dashboard must see. It is known only after
# the vendor responds, inside the tool, while the line is the engine's; the meter carries it across.


def test_metered_work_lands_on_the_running_tools_meter():
    with (
        tool_meter() as meter,
        metered(provider="xai", kind="image.generate", model="grok-imagine-image-2.0") as call,
    ):
        call.cost = 0.02

    assert meter.fields() == {
        "provider": "xai",
        "kind": "image.generate",
        "model": "grok-imagine-image-2.0",
        "count": None,
        "cost": 0.02,
        "cost_gap": None,
    }
    assert meter.failure is None


def test_a_metered_gap_names_why_a_computed_fee_is_missing_and_never_beside_a_fee():
    """Issue #880: the reason a tool could not price its work rides the meter to its `tool` line;
    a priced unit carries none, and a summed run is unpriced by its first unpriced unit's reason."""
    with (
        tool_meter() as meter,
        metered(provider="openai", kind="image.generate", model="gpt-image-9") as call,
    ):
        call.cost_gap = "unknown_model"
    assert meter.fields()["cost"] is None and meter.fields()["cost_gap"] == "unknown_model"

    with (
        tool_meter() as meter,
        metered(provider="openai", kind="image.generate", model="gpt-image-2.5-flare") as call,
    ):
        call.cost, call.cost_gap = ComputedCost(0.1), "unknown_model"
    assert meter.fields()["cost_gap"] is None  # a fee and a gap never ride together

    with tool_meter() as meter:
        meter_tool(provider="google", kind="search.grounding", model="g", cost=ComputedCost(0.01))
        meter_tool(provider="google", kind="search.grounding", model="g", cost_gap="not_reported")
    assert meter.fields()["cost"] is None and meter.fields()["cost_gap"] == "not_reported"


def test_a_block_that_never_prices_its_call_leaves_the_cost_absent():
    with (
        tool_meter() as meter,
        metered(provider="openai", kind="image.generate", model="gpt-image-2.5-flare"),
    ):
        pass

    assert meter.fields()["cost"] is None


def test_a_failed_unit_is_kept_on_the_meter_and_the_error_goes_through():
    """The tool turns the error into text for the model; the meter is how the line still says it
    failed, and what it was buying."""
    failure = RuntimeError("the API said no")
    with (
        tool_meter() as meter,
        pytest.raises(RuntimeError),
        metered(provider="openai", kind="image.generate", model="gpt-image-2.5-flare"),
    ):
        raise failure

    assert meter.failure is failure
    assert meter.fields()["provider"] == "openai" and meter.fields()["cost"] is None


def test_metered_work_outside_a_tool_run_records_nothing_and_raises_nothing():
    with metered(provider="openai", kind="audio.transcribe", model="gpt-transcribe") as call:
        call.cost = 0.01
    meter_tool(provider="google", kind="search.grounding", model="gemini-3.8-flash", count=1)
    with tool_meter() as meter:
        pass

    assert meter.fields() == {}


def test_several_units_are_summed_and_a_partial_total_is_never_stated():
    with tool_meter() as whole:
        meter_tool(
            provider="google", kind="search.grounding", model="m", count=2, cost=ComputedCost(0.028)
        )
        meter_tool(provider="google", kind="search.grounding", model="m", cost=ComputedCost(0.014))
    with tool_meter() as partial:
        meter_tool(provider="google", kind="search.grounding", model="m", cost=0.01)
        meter_tool(provider="google", kind="search.grounding", model="m")

    assert whole.fields()["count"] == 3
    assert whole.fields()["cost"] == pytest.approx(0.042)
    assert isinstance(whole.fields()["cost"], ComputedCost)
    assert partial.fields()["count"] == 2 and partial.fields()["cost"] is None


def test_a_failed_unit_adds_no_fee_and_a_partly_computed_total_says_computed():
    """A failed unit bought nothing, so the finished unit's fee stands; and a total any part of
    which is the harness's arithmetic is never presented as a figure the vendor stated."""
    with tool_meter() as meter:
        meter_tool(provider="xai", kind="image.edit", model="m", cost=0.02)
        with pytest.raises(RuntimeError), metered(provider="xai", kind="image.edit", model="m"):
            raise RuntimeError("the second edit failed")
    with tool_meter() as mixed:
        meter_tool(provider="openai", kind="image.generate", model="m", cost=0.02)
        meter_tool(provider="openai", kind="image.generate", model="m", cost=ComputedCost(0.03))

    assert meter.fields()["count"] == 2 and meter.fields()["cost"] == pytest.approx(0.02)
    assert isinstance(meter.failure, RuntimeError)
    assert isinstance(mixed.fields()["cost"], ComputedCost)
    assert mixed.fields()["cost"] == pytest.approx(0.05)


@pytest.mark.parametrize("cost", ["0.5", True, float("nan"), float("inf"), -0.01, object()])
def test_a_figure_that_is_not_a_fee_is_dropped_as_it_is_recorded(cost):
    with tool_meter() as meter:
        meter_tool(provider="xai", kind="image.generate", model="m", count="2", cost=cost)

    assert meter.fields()["cost"] is None and meter.fields()["count"] is None


class _SloppyTool(Tool):
    """An overlay tool that records its vendor's figure as the body spelled it."""

    name = "sloppy_media"
    description = "Buys an image and records a bad figure."

    def run(self, **_):
        with metered(provider="xai", kind="image.generate", model="m") as call:
            call.cost, call.count = "0.5", 2.0
        return "Generated and posted."


def test_a_bad_figure_from_an_overlay_tool_never_raises_into_the_wake(caplog):
    """The tool has already acted when its line is written: a figure it got wrong must cost the
    field, never the turn (the run's result still reaches the model)."""
    registry = ToolRegistry()
    registry.register(_SloppyTool())
    engine = Engine(provider=None, tools=registry)
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        assert engine._run_tool("sloppy_media", {}) == "Generated and posted."

    (line,) = tool_lines(caplog)
    assert line.startswith("tool name=sloppy_media executor=harness provider=xai ")
    assert " outcome=ok" in line and "cost=" not in line and "count=" not in line


# --- one cost shape on every line kind (issue #329) ------------------------------------------------

_COST_RE = re.compile(r"(?:^| )cost=([0-9.]+)(?: |$)")


def test_a_non_numeric_tool_cost_is_dropped_rather_than_crashing_the_wake(caplog):
    """A fee comes straight off a vendor body, so a surprise there costs the *field*, never the
    turn — the same guard the LLM cost has, since both render through `_money`."""
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        log_tool(name="grok_generate_image", executor="harness", outcome="ok", cost="free")  # type: ignore[arg-type]

    assert "cost=" not in caplog.records[0].getMessage()


def test_tool_and_llm_cost_render_in_the_same_plain_decimal_shape(caplog):
    """`cost=` must stay `cost=([0-9.]+)`-matchable — the same shape — on every line kind, and the
    ` llm provider=` head must never appear on a non-LLM line."""
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        log_llm_call(provider="xai", model="grok-4.3", seconds=1, cost=0.0038)
        log_tool(name="grok_generate_image", executor="harness", outcome="ok", cost=0.0038)

    llm, tool = (r.getMessage() for r in caplog.records[:2])
    assert _COST_RE.search(llm).group(1) == "0.0038"
    assert _COST_RE.search(tool).group(1) == "0.0038"  # identical, plain decimal, on both kinds
    # The head split the dashboard keys on: a tool line is never mistaken for an LLM line.
    assert tool.startswith("tool ") and not tool.startswith("llm ")


# --- a computed cost names its basis (issues #655, #656, #657) --------------
#
# OpenAI and Google state no price, so their cost is the harness's arithmetic over the vendor's
# published rates. The line says so with the fleet's existing literal, `cost_basis=computed` (the
# NOC's Steel launcher already writes it), immediately after `cost=` — and only there.


def test_a_computed_cost_is_followed_by_its_basis_on_the_llm_line(caplog):
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        log_llm_call(provider="openai", model="gpt-6-sol", seconds=1, cost=ComputedCost(0.0147))

    assert unstyled(caplog.records[0].getMessage()) == (
        "llm provider=openai purpose=main model=gpt-6-sol duration=1.00s cost=0.0147 "
        "cost_basis=computed outcome=ok"
    )


def test_a_stated_cost_carries_no_basis_so_its_line_is_unchanged(caplog):
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        log_llm_call(provider="xai", model="grok-4.7", seconds=1, cost=0.0147)

    assert "cost_basis" not in caplog.records[0].getMessage()


def test_a_basis_never_appears_without_the_figure_it_describes(caplog):
    """A computed cost the line drops (an unreported usage block's zero) takes its basis with it."""
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        log_llm_call(
            provider="google",
            model="gemini-3.8-flash",
            seconds=1,
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            cost=ComputedCost(0.0),
        )

    line = caplog.records[0].getMessage()
    assert "cost=" not in line and "cost_basis" not in line


def test_a_captured_computed_cost_keeps_its_basis_on_the_callers_line(caplog):
    """The describer and the reranker write their own line from what the adapter recorded; the
    basis travels with the figure, so neither can forget it."""
    with capture_llm_call() as call:
        log_llm_call(
            provider="google", model="gemini-3.8-flash", seconds=1, cost=ComputedCost(0.002)
        )
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        log_llm_call(
            provider=call.provider,
            purpose="helper",
            kind="image.describe",
            model=call.model,
            seconds=call.seconds,
            cost=call.cost,
            outcome="ok",
        )

    assert " cost=0.002 cost_basis=computed " in caplog.records[0].getMessage()


def test_computed_cost_arithmetic_is_a_plain_float():
    """Only a rate table makes one: a sum of two is a figure nobody has stood behind."""
    total = ComputedCost(0.01) + ComputedCost(0.02)
    assert type(total) is float and total == pytest.approx(0.03)


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
    line = kv(error="x." * 2500)

    assert len(line) < 300
    assert line.endswith("…")  # a bare (space-free) value needs no quoting — just the ellipsis

    # One unbroken token run past the bound is omitted, never cut: half a token is a credential
    # prefix no redaction rule can match (issue #787; `tests/test_log_value_cut.py`).
    assert kv(name="t", error="x" * 5000) == "name=t"

    # A prose error — the realistic case — is bounded *and* quoted, so the truncation can never
    # leave a dangling half-field for a parser to trip on.
    prose = kv(error="the tool exploded " * 100)
    assert len(prose) < 300
    assert prose.endswith('…"')


def test_a_bare_token_value_stays_unquoted_so_the_common_line_greps_cleanly():
    assert kv(timeline="019e7750-66ee-7f53-829f-13a8a710b6da", posted=3, duration="1.20s") == (
        "timeline=019e7750-66ee-7f53-829f-13a8a710b6da posted=3 duration=1.20s"
    )


# --- a control character in a value is a space (issue #806) ---------------------------------------

#: The fleet's shared witness: basecradle-router ``tests/vector/ai_otel_llm.yaml``, the case "a detail
#: renders by the fleet's one value rule, byte for byte with the router's log_fields" (router
#: ``307bcc9``, basecradle-router#396). Its input holds a ``"``, a ``'``, two backslashes, a newline,
#: an ESC and a C1 control character (``\x85``); the router's ``log_fields`` and its ``detail=``
#: remap both render it to `ROUTER_WITNESS_RENDERED`, and so must the harness, or the fleet has two
#: dialects of one grammar.
ROUTER_WITNESS_INPUT = "upstream said \" stage=wake outcome=failed duration=99999.0s\nit's C:\\path\\x\x1b[31m red\x85end"
ROUTER_WITNESS_RENDERED = (
    "\"upstream said ' stage=wake outcome=failed duration=99999.0s it's C:\\path\\x [31m red end\""
)


@pytest.mark.parametrize("render", [_value, _full_value])
def test_a_value_renders_the_routers_witness_byte_for_byte(render):
    assert render(ROUTER_WITNESS_INPUT) == ROUTER_WITNESS_RENDERED


@pytest.mark.parametrize(
    "control", ["\x00", "\x07", "\x1b", "\x7f", "\x80", "\x85", "\x9b", "\x9f"]
)
@pytest.mark.parametrize("render", [_value, _full_value])
def test_every_control_character_in_a_value_is_one_space(control, render):
    """C0, DEL and C1 alike: an ESC or a CSI (``\x9b``) quoted from a vendor's error body would paint
    the reader's terminal, a NUL or a BEL would mean something to it, and none is legitimate text."""
    rendered = render(f"before{control}after")
    assert rendered == '"before after"'


def test_a_run_of_control_characters_and_whitespace_is_one_space():
    assert _value("\x1b\x00 a\x1b\x1b\n\x85 b \x7f") == '"a b"'


def test_a_values_escape_is_removed_while_the_harness_paint_stays(monkeypatch):
    """The harness's own color is applied after a value is rendered, so the control rule strips what
    a value carried and never what `kv` or `head` painted."""
    monkeypatch.delenv(NO_COLOR_ENV, raising=False)
    line = head("wake end", BLUE) + " " + kv(outcome="error", error=f"vendor said {RED}boom{RESET}")

    assert line == (
        f'{BLUE}wake end{RESET} {RED}outcome=error{RESET} error="vendor said [31mboom [0m"'
    )
    assert line.count("\x1b") == 4  # the head's two and the verdict's two; none from the value


def test_the_unspoken_text_keeps_no_control_character(caplog):
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        log_unspoken("I will \x1b[2Jclear\x07 nothing", timeline="t")

    (record,) = caplog.records
    assert record.getMessage().endswith('text="I will [2Jclear nothing"')


# --- the ANSI verdict colors (issue #414) ------------------------------------
#
# The whole convention rests on one property: a color wraps a **whole token**, so nothing that
# could grep the plain line stops matching the colored one. These pin that property from both
# sides — the bytes that must be there, and the bytes that must *not* be inside a token.


def test_a_colored_head_keeps_its_token_contiguous():
    line = head("wake start", GREEN)

    assert line == f"{GREEN}wake start{RESET}"
    assert "wake start" in line  # the searchable bytes are still adjacent
    assert "\x1b" not in line[line.index("wake start") : line.index("wake start") + 10]


def test_every_fleet_color_is_the_sgr_code_the_convention_names():
    # Founder-decided 2026-08-17 and shared with the router and the NOC — a palette that differs
    # per repo is not a palette. Pinned as literals so a "tidy-up" cannot renumber them.
    assert (GREEN, BLUE, YELLOW, RED, RESET) == (
        "\x1b[32m",
        "\x1b[34m",
        "\x1b[33m",
        "\x1b[31m",
        "\x1b[0m",
    )


def test_a_verdict_pair_is_colored_whole_so_outcome_ok_still_greps():
    line = kv(timeline="abc", outcome="ok", posted=1)

    assert f"{GREEN}outcome=ok{RESET}" in line
    assert "outcome=ok" in line  # …and the pair itself is one contiguous run
    assert "timeline=abc" in line and "posted=1" in line


def test_each_outcome_word_gets_its_own_color():
    assert f"{GREEN}outcome=ok{RESET}" in kv(outcome="ok")
    assert f"{RED}outcome=error{RESET}" in kv(outcome="error")


def test_an_unrecognized_outcome_is_left_plain_rather_than_guessed():
    # The honest-absence rule `cost` and `endpoint` keep: a word outside the vocabulary is not
    # assigned a color on a hunch.
    assert kv(outcome="weird") == "outcome=weird"


def test_correlation_values_are_never_colored():
    # They are data a human copies out of the line and pastes into a query. An escape byte inside
    # one is a uuid that no longer round-trips.
    line = kv(timeline="019e7750-66ee-7f53-829f-13a8a710b6da", delivery="0199abc", outcome="ok")

    assert "timeline=019e7750-66ee-7f53-829f-13a8a710b6da" in line
    assert "delivery=0199abc" in line
    # Everything ahead of the first escape byte — i.e. everything but the verdict — is plain.
    assert (
        line.split("\x1b")[0] == "timeline=019e7750-66ee-7f53-829f-13a8a710b6da delivery=0199abc "
    )


def test_an_error_text_that_says_outcome_ok_is_not_colored_into_a_verdict():
    # `_value` already quotes it into a single `error=` value; the color layer keys on the *field*,
    # never on the text, so a hostile tool cannot paint itself green.
    line = kv(error="outcome=ok everything is fine")

    assert GREEN not in line
    assert line.startswith("error=")


def test_no_color_turns_the_whole_stream_plain(monkeypatch):
    monkeypatch.setenv(NO_COLOR_ENV, "1")

    assert color_enabled() is False
    assert head("wake failed", RED) == "wake failed"
    assert kv(timeline="abc", outcome="error") == "timeline=abc outcome=error"


def test_an_empty_no_color_is_not_an_opt_out(monkeypatch):
    # https://no-color.org: *present and non-empty*. An exported-but-blank var is how a shell
    # leaves a variable it never set, and it must not silently disable the feature.
    monkeypatch.setenv(NO_COLOR_ENV, "")

    assert color_enabled() is True
    assert head("wake start", GREEN) == f"{GREEN}wake start{RESET}"


def test_the_color_setting_is_read_per_line_not_frozen_at_import(monkeypatch):
    # `agent.env` is loaded after this module imports, so a value cached at import would answer for
    # the wrong environment.
    assert head("wake start", GREEN).startswith(GREEN)
    monkeypatch.setenv(NO_COLOR_ENV, "1")
    assert head("wake start", GREEN) == "wake start"
    monkeypatch.delenv(NO_COLOR_ENV)
    assert head("wake start", GREEN).startswith(GREEN)


def test_the_fact_heads_stay_plain_so_the_dashboards_extraction_anchors_survive(caplog):
    # `llm` and `unspoken` name a *fact*, not a verdict — and the fleet dashboard extracts on their
    # heads with the following field attached (` llm provider=`, ` unspoken timeline=`). A head span
    # would sit exactly in that gap. (`tool` is the same, and is the next test: its line carries a
    # verdict field too.)
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        log_llm_call(provider="openai", model="gpt-5.4-mini", seconds=1.0)
        log_unspoken("thinking out loud", timeline="019e77")

    lines = [r.getMessage() for r in caplog.records]
    # The heads are plain; the one colored span is the `outcome=` verdict, wrapped whole, which
    # every `llm` line carries since issue #782.
    assert "\x1b" not in "".join(m.replace(f"{GREEN}outcome=ok{RESET}", "") for m in lines)
    assert any(m.startswith("llm provider=") for m in lines)
    assert any(m.startswith("unspoken timeline=") for m in lines)


def test_the_tool_lines_outcome_is_colored_but_its_head_is_not(caplog):
    # The one line carrying both halves of the split: ` tool name=` is a dashboard anchor and stays
    # plain, while its `outcome=` is a verdict and does not.
    started = datetime(2026, 8, 17, tzinfo=UTC)
    engine = Engine(provider=None, tools=ToolRegistry(), clock=lambda: started)

    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        engine._log_tool("memory", started)
        engine._log_tool("memory", started, reason="exception", error="boom")

    lines = [r.getMessage() for r in caplog.records]
    assert all(m.startswith("tool name=memory") for m in lines)
    assert any(f"{GREEN}outcome=ok{RESET}" in m for m in lines)
    assert any(f"{RED}outcome=error{RESET}" in m for m in lines)
    # …and the dashboard's own two-token pattern still spans the line.
    assert any(re.search(r"tool name=.*outcome=error", m) for m in lines)


def test_a_colored_head_is_followed_by_the_reset_and_not_by_its_separator():
    """The precise byte shape of a colored line — and the one thing the token rule does NOT buy.

    A whole-token search keeps working (`wake end`, `outcome=error`). A pattern that reaches *past*
    the token — a trailing-space anchor (``wake failed ``) or an adjacent-pair literal
    (``wake reported_failure kind=billing``) — does not, because the head's ``RESET`` now sits in
    that gap. This is the shape every out-of-repo consumer of these lines is coupled to, so it is
    pinned here rather than left to be rediscovered: a pattern spanning the gap must wildcard it
    (``wake end.*outcome=error``), never spell it as a literal space.
    """
    line = f"{head('wake end', BLUE)} {kv(timeline='abc', outcome='error')}"

    assert line == f"{BLUE}wake end{RESET} timeline=abc {RED}outcome=error{RESET}"
    # Whole tokens: still searchable, exactly as the convention promises.
    assert "wake end" in line and "outcome=error" in line
    assert re.search(r"wake end.*outcome=error", line)
    # Across the gap: no longer. Both shapes below are live NOC patterns as of this change.
    assert not re.search(r"wake end .*outcome=error", line)
    assert not re.search(r"wake end timeline=", line)


# === issue #488: a zero is not a measurement, and a fragment is not an answer =


def _line(caplog):
    return next(r.getMessage() for r in caplog.records if r.getMessage().startswith("llm "))


def test_a_usage_block_of_nothing_but_zeros_is_not_usage():
    """The live signature of a broken stream: an answer arrived, the vendor counted nothing."""
    assert usage_reported({"prompt_tokens": 812, "completion_tokens": 96}) is True
    assert usage_reported({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}) is False
    assert usage_reported(None) is False
    # One non-zero count is a report: a cached-prompt hit on a tiny call is a real measurement.
    assert usage_reported({"prompt_tokens": 0, "completion_tokens": 4}) is True


def test_an_all_zero_usage_block_is_omitted_rather_than_printed_as_zeros(caplog):
    """`tokens_in=0 … cost=0 outcome=ok` reads on the dashboard as a free call that worked.

    It was neither. The line the live describer wrote said exactly that beside a description cut
    mid-sentence, which is how a broken stream passed for a success (issue #488).
    """
    with caplog.at_level(logging.DEBUG, logger="basecradle_harness"):
        log_llm_call(
            provider="openrouter",
            model="google/gemini-3.8-flash",
            seconds=12.26,
            usage={
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "completion_tokens_details": {"reasoning_tokens": 0},
            },
            cost=0.0,
            outcome="ok",
        )

    line = _line(caplog)
    assert "tokens_" not in line and "cost=" not in line
    # What the vendor *did* say still rides the line — the absence is only of what it did not.
    assert "provider=openrouter" in line and "duration=12.26s" in line


def test_a_genuinely_free_call_still_reports_its_zero_cost(caplog):
    """The other half of the rule: a `:free` endpoint counts real tokens, so `cost=0` is a fact."""
    with caplog.at_level(logging.DEBUG, logger="basecradle_harness"):
        log_llm_call(
            provider="openrouter",
            model="z-ai/glm-5.2:free",
            seconds=1.0,
            usage={"prompt_tokens": 900, "completion_tokens": 12},
            cost=0.0,
            outcome="ok",
        )

    line = _line(caplog)
    assert "tokens_in=900" in line and "cost=0" in line


def test_a_real_charge_survives_a_usage_block_that_reported_nothing(caplog):
    """The dollar is dropped as an *absence*, never as a rule about zeros.

    The native xAI adapter reads its cost off the **response** (`cost_usd`), not off usage — so a
    real charge beside an unreported usage block is an independent claim, and swallowing it would
    lose spend from the rollup to fix a line that was only ever wrong about zeros.
    """
    with caplog.at_level(logging.DEBUG, logger="basecradle_harness"):
        log_llm_call(
            provider="xai",
            model="grok-4.3",
            seconds=1.0,
            usage={"prompt_tokens": 0, "completion_tokens": 0},
            cost=0.004,
            outcome="ok",
        )

    line = _line(caplog)
    assert "cost=0.004" in line and "tokens_" not in line


def test_a_usage_the_adapter_never_reported_leaves_the_cost_alone(caplog):
    """``usage=None`` is *nothing was stated*, which is not the same claim as *zero*.

    An adapter that reports a dollar figure without a usage block (the xAI SDK reads its cost off
    the response, not off usage) is stating something it knows, and it must survive.
    """
    with caplog.at_level(logging.DEBUG, logger="basecradle_harness"):
        log_llm_call(provider="xai", model="grok-4.3", seconds=1.0, cost=0.004, outcome="ok")

    assert "cost=0.004" in _line(caplog)


def test_the_finish_reason_is_read_off_every_shape_a_shipped_adapter_hands_back():
    """One reader, three wires — `serving_endpoint`'s shape, for the same reason (issue #488)."""

    class _Native:  # the native xai-sdk Response: a property naming its proto enum
        finish_reason = "REASON_MAX_LEN"

    assert finish_reason({"choices": [{"finish_reason": "length"}]}) == "length"
    assert finish_reason({"choices": [{"finish_reason": "stop"}]}) == "stop"
    assert (
        finish_reason(
            {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}}
        )
        == "max_output_tokens"
    )
    assert finish_reason(_Native()) == "REASON_MAX_LEN"
    # A Responses payload that completed states no finish reason, and neither does an empty one.
    assert finish_reason({"status": "completed", "output": []}) is None
    assert finish_reason({}) is None and finish_reason(None) is None


def test_reading_a_finish_reason_never_breaks_a_turn():
    """Observability's standing rule, applied to a vendor object that raises on attribute access."""

    class _Hostile:
        @property
        def choices(self):
            raise RuntimeError("the SDK changed under us")

    assert finish_reason(_Hostile()) is None


def test_every_vendors_spelling_of_out_of_room_is_recognized():
    for spelling in ("length", "LENGTH", "max_output_tokens", "max_tokens", "REASON_MAX_LEN"):
        assert truncated(spelling), spelling
    for spelling in ("stop", "tool_calls", "REASON_STOP", "content_filter", None, ""):
        assert not truncated(spelling), spelling


def test_a_captured_call_carries_the_finish_reason_but_the_line_never_prints_it(caplog):
    """It is evidence for the caller, not a field on a byte-frozen grammar (issue #488)."""
    with caplog.at_level(logging.DEBUG, logger="basecradle_harness"), capture_llm_call() as call:
        log_llm_call(provider="openrouter", model="m", seconds=1.0, finish_reason="length")

    assert call.finish_reason == "length"
    assert not [r for r in caplog.records if r.getMessage().startswith("llm ")]


# === issue #759: one meaning per token field, and the refusal head ==============================


def test_reasoning_reported_apart_is_folded_into_the_output_it_was_billed_as():
    """xAI's own example: 32 prompt + 9 completion + 94 reasoning = 135 (docs.x.ai)."""
    from basecradle_harness._observability import reasoning_apart

    wire = reasoning_apart(
        {
            "prompt_tokens": 32,
            "completion_tokens": 9,
            "total_tokens": 135,
            "prompt_tokens_details": {"cached_tokens": 6},
            "completion_tokens_details": {"reasoning_tokens": 94},
        }
    )
    assert token_counts(wire) == {
        "tokens_in": 32,
        "tokens_out": 103,
        "tokens_total": 135,
        "cached_tokens": 6,
        "tokens_reasoning": 94,
    }
    assert reasoning_apart(None) is None
    # No reasoning reported: the counts are the wire's own.
    plain = reasoning_apart({"prompt_tokens": 10, "completion_tokens": 4})
    assert token_counts(plain) == {"tokens_in": 10, "tokens_out": 4, "tokens_total": 14}


def test_the_total_is_the_sum_of_its_two_fields_never_the_vendors_own():
    counts = token_counts({"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 99})
    assert counts["tokens_total"] == 14
    # Half a count is no total: an honest absence, never a sum with a hole in it.
    assert "tokens_total" not in token_counts({"prompt_tokens": 10, "total_tokens": 10})


def test_a_refusal_is_the_calls_own_outcome_and_a_capture_keeps_it_for_its_caller(caplog):
    """One refused attempt, one line (issue #782): ``outcome=refused`` with the vendor's category.

    Inside a capture the adapter's line is held back and the refusal rides the record, so the
    caller's one line can say it.
    """
    from basecradle_harness._observability import HELPER, capture_llm_call

    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        log_llm_call(
            provider="anthropic",
            model="claude-haiku-5-5",
            seconds=1.0,
            outcome="refused",
            category="Cyber Risk",
            level=logging.WARNING,
        )
        with capture_llm_call(HELPER) as call:
            log_llm_call(
                provider="anthropic", model="claude-haiku-5-5", seconds=1.0, outcome="refused"
            )
    (line,) = [unstyled(r.getMessage()) for r in caplog.records]
    assert line == (
        "llm provider=anthropic purpose=main model=claude-haiku-5-5 duration=1.00s "
        "outcome=refused category=cyber_risk"
    )
    assert caplog.records[0].levelno == logging.WARNING
    assert (call.outcome, call.category) == ("refused", None)


def test_a_billed_answer_the_parser_trips_on_is_still_billed_once(router, provider, caplog):
    """An answer malformed in a shape the parser never foresaw raises whatever it trips on, and was
    billed all the same: its line is still written, `fallback reason=invalid_response`, with its
    tokens, and the original exception is what propagates (issue #782 review)."""
    import httpx

    from basecradle_harness import Message
    from tests.conftest import CHAT_URL

    body = {
        "id": "chatcmpl-0123456789abcdef",
        "object": "chat.completion",
        "created": 1791030923,
        "model": "gpt-5.4-mini",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "call_0", "type": "function", "function": None}],
                },
            }
        ],
        "usage": {"prompt_tokens": 90, "completion_tokens": 10, "total_tokens": 100},
    }
    router.post(CHAT_URL).mock(return_value=httpx.Response(200, json=body))
    with (
        caplog.at_level(logging.INFO, logger="basecradle_harness"),
        pytest.raises(AttributeError),
    ):
        provider.chat([Message.user("hi")])
    (line,) = [
        unstyled(r.getMessage()) for r in caplog.records if r.getMessage().startswith("llm ")
    ]
    assert " tokens_in=90 " in line
    assert " outcome=fallback reason=invalid_response " in line
