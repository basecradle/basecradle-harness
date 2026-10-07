"""OpenAI's cost, computed from its published rates (issue #657).

OpenAI states tokens and no price, so the harness prices the call from `_openai_rates`. These pin the
table's arithmetic against hand-worked figures, every case where the honest answer is *no figure*,
and the lines the adapter and the media tools write: ``cost=`` tagged ``cost_basis=computed`` on the
``llm`` line, a priced ``media`` line per web search, a stated gap for containers, and nothing at all
computed for an endpoint that is not OpenAI.
"""

import logging
import re

import httpx
import pytest

from basecradle_harness import Message, OpenAIProvider
from basecradle_harness._observability import ComputedCost
from basecradle_harness._openai_rates import (
    CONTAINER_GAP,
    RATES,
    SOURCE,
    Usage,
    call_cost,
    image_cost,
    known,
    priced_host,
    ran_container,
    search_calls,
    transcription_cost,
    usage_of,
    web_search_cost,
)
from tests.conftest import (
    FAKE_KEY,
    out_code_interpreter_call,
    out_message,
    out_web_search_call,
    responses_body,
)

OPENAI = "https://api.openai.com/v1"


def _approx(value):
    return pytest.approx(value, rel=1e-12)


# --- the table ---------------------------------------------------------------------------------


def test_the_source_names_the_page_and_the_date_it_was_read():
    assert SOURCE.startswith("https://developers.openai.com/api/docs/pricing ")
    assert "read 2026-10-07" in SOURCE


def test_every_model_the_fleet_has_run_is_priced():
    """gpt-6-sol today; gpt-5.6-terra and gpt-5.4-mini before it (the NOC inventory's history)."""
    for model in ("gpt-6-sol", "gpt-5.6-terra", "gpt-5.4-mini"):
        assert known(model), model


def test_a_gpt_6_sol_call_is_priced_at_its_three_input_rates_and_its_output_rate():
    """10,000 in (6,000 cached, 1,000 written to the cache) + 500 out, short context.

    3,000 × $2.00 + 6,000 × $0.20 + 1,000 × $2.50 + 500 × $10.00 = $0.0147 per million → $0.0147.
    """
    usage = Usage(input=10_000, cached=6_000, cache_write=1_000, output=500)
    cost = call_cost("gpt-6-sol", usage)
    assert isinstance(cost, ComputedCost)
    assert cost == _approx((3_000 * 2.00 + 6_000 * 0.20 + 1_000 * 2.50 + 500 * 10.00) / 1e6)


def test_above_272k_every_token_of_the_call_is_priced_on_the_long_row():
    usage = Usage(input=300_000, cached=200_000, output=1_000)
    assert call_cost("gpt-6-sol", usage) == _approx(
        (100_000 * 4.00 + 200_000 * 0.40 + 1_000 * 15.00) / 1e6
    )


def test_exactly_272k_is_still_short_context():
    usage = Usage(input=272_000, cached=0, output=0)
    assert call_cost("gpt-6-sol", usage) == _approx(272_000 * 2.00 / 1e6)


def test_a_model_without_a_long_row_is_not_priced_above_the_threshold():
    assert RATES["gpt-5.4-mini"].long is None
    assert call_cost("gpt-5.4-mini", Usage(input=272_001, cached=0, output=1)) is None
    assert call_cost("gpt-5.4-mini", Usage(input=1_000, cached=0, output=1)) is not None


def test_a_model_with_no_cache_write_rate_prices_a_written_token_as_ordinary_input():
    """Before GPT-5.6 there is "no additional cache-write charge"."""
    with_write = call_cost("gpt-5.4", Usage(input=1_000, cached=0, cache_write=400, output=0))
    without = call_cost("gpt-5.4", Usage(input=1_000, cached=0, output=0))
    assert with_write == without == _approx(1_000 * 2.50 / 1e6)


def test_a_dated_snapshot_is_priced_as_its_alias():
    usage = Usage(input=1_000, cached=0, output=100)
    assert call_cost("gpt-6-sol-2026-08-14", usage) == call_cost("gpt-6-sol", usage)


def test_an_unknown_model_is_not_priced():
    assert not known("gpt-7-nova")
    assert call_cost("gpt-7-nova", Usage(input=1_000, cached=0, output=100)) is None


@pytest.mark.parametrize("served", ["flex", "priority", "fast", "ultrafast", "scale"])
def test_a_call_served_on_any_tier_but_standard_is_not_priced(served):
    assert (
        call_cost("gpt-6-sol", Usage(input=1_000, cached=0, output=1), served_tier=served) is None
    )


def test_a_response_that_names_no_tier_is_standard_only_if_the_request_asked_for_none():
    usage = Usage(input=1_000, cached=0, output=1)
    assert call_cost("gpt-6-sol", usage, served_tier="default") is not None
    assert call_cost("gpt-6-sol", usage, served_tier=None, asked_tier=None) is not None
    assert call_cost("gpt-6-sol", usage, served_tier=None, asked_tier="auto") is not None
    assert call_cost("gpt-6-sol", usage, served_tier=None, asked_tier="flex") is None


def test_the_usage_block_is_read_from_either_surface():
    responses = {
        "input_tokens": 10,
        "output_tokens": 3,
        "input_tokens_details": {"cached_tokens": 4, "cache_write_tokens": 2},
    }
    chat = {
        "prompt_tokens": 10,
        "completion_tokens": 3,
        "prompt_tokens_details": {"cached_tokens": 4, "cache_write_tokens": 2},
    }
    expected = Usage(input=10, cached=4, cache_write=2, output=3)
    assert usage_of(responses) == usage_of(chat) == expected
    assert usage_of(None) is None
    assert usage_of({"total_tokens": 9}) is None


def test_only_openais_own_host_is_priced():
    assert priced_host(None)
    assert priced_host("https://api.openai.com/v1")
    assert not priced_host("https://eu.api.openai.com/v1")  # data residency: a 10% uplift
    assert not priced_host("https://api.openai.test/v1")


# --- media and tools ---------------------------------------------------------------------------


def test_an_image_is_priced_by_its_image_and_text_tokens():
    usage = {
        "input_tokens": 50,
        "input_tokens_details": {"image_tokens": 0, "text_tokens": 50},
        "output_tokens": 4_160,
        "total_tokens": 4_210,
    }
    assert image_cost("gpt-image-2.5-flare", usage) == _approx((50 * 5.00 + 4_160 * 30.00) / 1e6)


def test_an_edit_prices_its_source_images_at_the_image_input_rate():
    usage = {
        "input_tokens": 1_050,
        "input_tokens_details": {"image_tokens": 1_000, "text_tokens": 50},
        "output_tokens": 4_160,
        "output_tokens_details": {"image_tokens": 4_160, "text_tokens": 0},
    }
    assert image_cost("gpt-image-2.5-sunburst", usage) == _approx(
        (1_000 * 8.00 + 50 * 5.00 + 4_160 * 30.00) / 1e6
    )


def test_an_image_call_without_its_input_split_or_with_unpriced_text_output_is_not_priced():
    assert image_cost("gpt-image-2.5-flare", {"input_tokens": 50, "output_tokens": 10}) is None
    text_out = {
        "input_tokens_details": {"image_tokens": 0, "text_tokens": 50},
        "output_tokens": 10,
        "output_tokens_details": {"image_tokens": 0, "text_tokens": 10},
    }
    assert image_cost("gpt-image-2.5-flare", text_out) is None
    assert image_cost("gpt-image-9", text_out) is None


def test_a_transcription_is_priced_per_minute_or_per_token_as_its_usage_says():
    assert transcription_cost("gpt-transcribe", {"type": "duration", "seconds": 90}) == _approx(
        1.5 * 0.0045
    )
    tokens = {"type": "tokens", "input_tokens": 1_000, "output_tokens": 200}
    assert transcription_cost("gpt-4o-transcribe", tokens) == _approx(
        (1_000 * 2.50 + 200 * 10.00) / 1e6
    )
    assert transcription_cost("gpt-transcribe", tokens) is None  # no token rate stated
    assert transcription_cost("gpt-transcribe", None) is None


def test_only_a_search_action_is_a_billed_web_search_call():
    output = [
        out_web_search_call(query="a"),
        {
            "type": "web_search_call",
            "id": "ws-2",
            "status": "completed",
            "action": {"type": "open_page"},
        },
        out_web_search_call(query="b"),
        out_message("done"),
    ]
    assert search_calls(output) == 2
    assert web_search_cost(2) == _approx(0.02)
    assert web_search_cost(0) is None


def test_a_container_run_is_recognized():
    assert ran_container([out_code_interpreter_call(code="print(1)")])
    assert not ran_container([out_message("hi")])


# --- the adapter's lines -----------------------------------------------------------------------


def _usage(**overrides):
    usage = {
        "input_tokens": 10_000,
        "output_tokens": 500,
        "total_tokens": 10_500,
        "input_tokens_details": {"cached_tokens": 6_000, "cache_write_tokens": 1_000},
        "output_tokens_details": {"reasoning_tokens": 100},
    }
    usage.update(overrides)
    return usage


def _body(*output, usage=None, service_tier="default", model="gpt-6-sol"):
    body = responses_body(*output)
    body.update(model=model, usage=usage or _usage(), service_tier=service_tier)
    return body


@pytest.fixture
def sol():
    provider = OpenAIProvider(model="gpt-6-sol", api_key=FAKE_KEY, surface="responses")
    yield provider
    provider.close()


def _lines(caplog, head):
    return [m for m in (r.getMessage() for r in caplog.records) if m.startswith(head)]


def test_a_gpt_6_sol_brain_call_logs_a_computed_cost(router, sol, caplog):
    router.post(f"{OPENAI}/responses").mock(
        return_value=httpx.Response(200, json=_body(out_message("hi")))
    )
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        sol.chat([Message.user("go")])
    (line,) = _lines(caplog, "llm ")
    assert re.fullmatch(
        r"llm provider=openai purpose=main model=gpt-6-sol duration=\d+\.\d\ds tokens_in=10000 "
        r"tokens_out=500 tokens_total=10500 cached_tokens=6000 cost=0\.0147 cost_basis=computed "
        r"generation_id=resp-fake0001",
        line,
    ), line
    assert not _lines(caplog, "media ")


def test_a_web_search_gets_its_own_priced_media_line(router, sol, caplog):
    output = (out_web_search_call(query="a"), out_web_search_call(query="b"), out_message("x"))
    router.post(f"{OPENAI}/responses").mock(return_value=httpx.Response(200, json=_body(*output)))
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        sol.chat([Message.user("search")])
    assert _lines(caplog, "media ") == [
        "media provider=openai kind=search.web model=gpt-6-sol count=2 cost=0.02 cost_basis=computed"
    ]


def test_a_container_is_a_stated_gap_once_per_wake(router, sol, caplog):
    output = (out_code_interpreter_call(code="print(1)"), out_message("1"))
    router.post(f"{OPENAI}/responses").mock(return_value=httpx.Response(200, json=_body(*output)))
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        sol.chat([Message.user("run")])
        sol.chat([Message.user("again")])
    gaps = [r for r in caplog.records if r.getMessage() == CONTAINER_GAP]
    assert len(gaps) == 1 and gaps[0].levelno == logging.WARNING


def test_an_unknown_model_logs_no_cost_and_warns_once(router, caplog):
    provider = OpenAIProvider(model="gpt-7-nova", api_key=FAKE_KEY, surface="responses")
    router.post(f"{OPENAI}/responses").mock(
        return_value=httpx.Response(200, json=_body(out_message("hi"), model="gpt-7-nova"))
    )
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        provider.chat([Message.user("go")])
        provider.chat([Message.user("go")])
    lines = _lines(caplog, "llm ")
    assert len(lines) == 2 and all(" cost=" not in line for line in lines)
    warnings = [r for r in caplog.records if "No published rate" in r.getMessage()]
    assert len(warnings) == 1 and warnings[0].levelno == logging.WARNING


def test_a_flex_call_logs_no_cost(router, sol, caplog):
    router.post(f"{OPENAI}/responses").mock(
        return_value=httpx.Response(200, json=_body(out_message("hi"), service_tier="flex"))
    )
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        sol.chat([Message.user("go")])
    (line,) = _lines(caplog, "llm ")
    assert " cost=" not in line and "cost_basis" not in line


def test_another_host_is_never_priced_and_says_so_once(router, caplog):
    provider = OpenAIProvider(
        model="gpt-6-sol",
        api_key=FAKE_KEY,
        base_url="https://eu.api.openai.com/v1",
        surface="responses",
    )
    output = (out_web_search_call(), out_message("x"))
    router.post("https://eu.api.openai.com/v1/responses").mock(
        return_value=httpx.Response(200, json=_body(*output))
    )
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        provider.chat([Message.user("go")])
        provider.chat([Message.user("go")])
    assert all(" cost=" not in line for line in _lines(caplog, "llm "))
    assert (
        _lines(caplog, "media ")[0]
        == "media provider=openai kind=search.web model=gpt-6-sol count=1"
    )
    assert sum("not api.openai.com" in r.getMessage() for r in caplog.records) == 1


def test_an_xai_endpoint_behind_the_openai_sdk_is_never_priced_from_openais_table(router, caplog):
    """The label decides whose rates could apply: OpenAI's say nothing about xAI's."""
    provider = OpenAIProvider(
        model="gpt-6-sol",
        api_key=FAKE_KEY,
        base_url="https://api.x.ai/v1",
        provider="xai",
        surface="responses",
    )
    output = (out_web_search_call(), out_message("x"))
    router.post("https://api.x.ai/v1/responses").mock(
        return_value=httpx.Response(200, json=_body(*output))
    )
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        provider.chat([Message.user("go")])
    assert " cost=" not in _lines(caplog, "llm ")[0]
    assert not _lines(caplog, "media ")
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_a_vendor_stated_cost_wins_and_carries_no_basis(router, caplog):
    """OpenRouter behind the openai SDK states its own dollars; nothing is computed over them."""
    provider = OpenAIProvider(
        model="z-ai/glm-5.3",
        api_key=FAKE_KEY,
        base_url="https://openrouter.ai/api/v1",
        provider="openrouter",
        surface="responses",
    )
    router.post("https://openrouter.ai/api/v1/responses").mock(
        return_value=httpx.Response(
            200, json=_body(out_message("x"), usage=_usage(cost=0.0123), model="z-ai/glm-5.3")
        )
    )
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        provider.chat([Message.user("go")])
    line = _lines(caplog, "llm ")[0]
    assert " cost=0.0123 " in line and "cost_basis" not in line


def test_an_image_breakdown_naming_one_half_has_the_other_as_the_remainder():
    usage = {
        "input_tokens_details": {"image_tokens": 0, "text_tokens": 50},
        "output_tokens": 4_160,
        "output_tokens_details": {"text_tokens": 0},
    }
    assert image_cost("gpt-image-2.5-flare", usage) == _approx((50 * 5.00 + 4_160 * 30.00) / 1e6)


def test_an_image_breakdown_that_does_not_add_up_is_not_priced():
    """Dropping the unexplained tokens would drop $30/M output from a figure tagged computed."""
    usage = {
        "input_tokens_details": {"image_tokens": 0, "text_tokens": 50},
        "output_tokens": 4_160,
        "output_tokens_details": {"image_tokens": 1_000, "text_tokens": 0},
    }
    assert image_cost("gpt-image-2.5-flare", usage) is None


def test_a_known_model_on_a_call_the_table_cannot_price_is_a_stated_gap(router, caplog):
    """A Flex call, and a long prompt on a model with no long-context row: one WARNING each."""
    provider = OpenAIProvider(model="gpt-5.4-mini", api_key=FAKE_KEY, surface="responses")
    long_prompt = _usage(input_tokens=300_000, total_tokens=300_500)
    router.post(f"{OPENAI}/responses").mock(
        side_effect=[
            httpx.Response(200, json=_body(out_message("a"), service_tier="flex")),
            httpx.Response(200, json=_body(out_message("b"), service_tier="flex")),
            httpx.Response(200, json=_body(out_message("c"), usage=long_prompt)),
        ]
    )
    with caplog.at_level(logging.INFO, logger="basecradle_harness"):
        for _ in range(3):
            provider.chat([Message.user("go")])
    gaps = [r.getMessage() for r in caplog.records if "not priced (service_tier=" in r.getMessage()]
    assert len(gaps) == 2
    assert "service_tier=flex" in gaps[0] and "input_tokens=300000" in gaps[1]
    assert all(" cost=" not in line for line in _lines(caplog, "llm "))
