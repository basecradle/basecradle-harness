"""Anthropic's published rates as the harness transcribed them (`_anthropic_rates`, issue #725).

Every figure pinned here is the page's own number multiplied out by hand, never the table read back
through the function under test — a test that derives its expectation from the table proves only
that the table agrees with itself.
"""

from __future__ import annotations

import pytest

from basecradle_harness._anthropic_rates import (
    HAIKU_LONG_PROMPT_THRESHOLD,
    Usage,
    call_cost,
    known,
    server_tool_requests,
    usage_of,
    web_search_cost,
)
from basecradle_harness._observability import ComputedCost


def _usage(uncached=0, write_5m=0, write_1h=0, read=0, output=0) -> Usage:
    return Usage(uncached=uncached, write_5m=write_5m, write_1h=write_1h, read=read, output=output)


def test_haiku_5_5_up_to_100k_prices_every_category_on_the_short_row():
    # $0.10 in, $0.125 5m write, $0.20 1h write, $0.01 read, $0.50 out — per million.
    cost = call_cost("claude-haiku-5-5", _usage(10_000, 20_000, 30_000, 40_000, 5_000))
    assert isinstance(cost, ComputedCost)
    assert cost == pytest.approx((1_000 + 2_500 + 6_000 + 400 + 2_500) / 1_000_000)


def test_haiku_5_5_over_100k_prices_every_token_on_the_long_row_cache_reads_counted():
    """ "A request's prompt length counts all of its input tokens, including cache reads and writes.\""""
    over = _usage(uncached=1, read=HAIKU_LONG_PROMPT_THRESHOLD, output=1_000)
    # $0.50 in, $0.05 read, $2.50 out.
    assert call_cost("claude-haiku-5-5", over) == pytest.approx(
        (1 * 0.50 + 100_000 * 0.05 + 1_000 * 2.50) / 1_000_000
    )
    at = _usage(read=HAIKU_LONG_PROMPT_THRESHOLD, output=1_000)  # exactly 100,000: not over
    assert call_cost("claude-haiku-5-5", at) == pytest.approx(
        (100_000 * 0.01 + 1_000 * 0.50) / 1_000_000
    )


def test_a_flat_model_is_priced_on_its_one_row_at_any_length():
    big = _usage(uncached=500_000, write_1h=1_000, read=200_000, output=10_000)
    # Opus 5.5: $4 in, $8 1h write, $0.20 read, $20 out.
    assert call_cost("claude-opus-5-5", big) == pytest.approx(
        (500_000 * 4 + 1_000 * 8 + 200_000 * 0.20 + 10_000 * 20) / 1_000_000
    )


def test_us_inference_is_1_1x_and_any_other_geography_is_not_priced():
    usage = _usage(uncached=1_000_000)
    assert call_cost("claude-sonnet-5-5", usage, geo="us") == pytest.approx(2.0 * 1.1)
    assert call_cost("claude-sonnet-5-5", usage, geo="global") == pytest.approx(2.0)
    assert call_cost("claude-sonnet-5-5", usage, geo="eu") is None


@pytest.mark.parametrize(("tier", "speed"), [("priority", None), ("batch", None), (None, "fast")])
def test_a_tier_or_speed_off_the_standard_row_is_not_priced(tier, speed):
    assert call_cost("claude-opus-5-5", _usage(uncached=10), tier=tier, speed=speed) is None


def test_an_unknown_model_is_not_priced_and_a_snapshot_is_its_alias():
    assert call_cost("claude-future-9", _usage(uncached=10)) is None
    assert not known("claude-future-9")
    assert known("claude-haiku-4-5-20251001")
    assert call_cost("claude-haiku-4-5-20251001", _usage(uncached=1_000_000)) == pytest.approx(1.0)


def test_web_search_is_ten_dollars_a_thousand():
    assert web_search_cost(3) == pytest.approx(0.03)
    assert isinstance(web_search_cost(3), ComputedCost)
    assert web_search_cost(0) is None


def test_usage_splits_writes_by_the_breakdown_or_by_the_ttl_asked_for():
    reported = {
        "input_tokens": 5,
        "output_tokens": 6,
        "cache_creation_input_tokens": 300,
        "cache_read_input_tokens": 70,
        "cache_creation": {"ephemeral_5m_input_tokens": 100, "ephemeral_1h_input_tokens": 200},
    }
    assert usage_of(reported) == _usage(5, 100, 200, 70, 6)
    bare = {key: value for key, value in reported.items() if key != "cache_creation"}
    assert usage_of(bare, written_ttl="1h") == _usage(5, 0, 300, 70, 6)
    assert usage_of(bare) == _usage(5, 300, 0, 70, 6)
    assert usage_of({"cache_read_input_tokens": 1}) is None


def test_server_tool_requests_reads_declared_and_undeclared_fields():
    assert (
        server_tool_requests({"server_tool_use": {"web_search_requests": 4}}, "web_search_requests")
        == 4
    )
    assert server_tool_requests({}, "web_search_requests") == 0

    class Block:  # an SDK model that kept a field it was not told about
        model_extra = {"code_execution_requests": 2}

    class Usage_:
        server_tool_use = Block()

    assert server_tool_requests(Usage_(), "code_execution_requests") == 2
