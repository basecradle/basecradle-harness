"""The Vertex rate table (`_google_rates`) — where the harness computes a Gemini call's dollars.

Every expected figure below is worked by hand from Google's published Standard rows, so a test reads
as the arithmetic it pins. The rule under all of them: a call the table cannot price gets ``None``,
never an estimate (issue #655).
"""

from __future__ import annotations

import datetime
import itertools
from types import SimpleNamespace

import pytest

from basecradle_harness._google_rates import (
    GROUNDING,
    LONG_CONTEXT_THRESHOLD,
    RATES,
    Usage,
    billing_day,
    call_cost,
    from_usage_metadata,
    grounding_cost,
    known,
)
from basecradle_harness._observability import ComputedCost

INTRO = datetime.date(2026, 10, 6)
STANDARD = datetime.date(2027, 1, 1)


def cost(model="gemini-3.8-flash", location="us", on=INTRO, **usage):
    fields = {"prompt": 1_000_000, "cached": 0, "output": 0, **usage}
    return call_cost(model, location, Usage(**fields), on=on)


def test_the_us_multi_region_is_priced_as_non_global():
    """Google prices every location but ``global`` 10% higher on its GA Gemini 3 families."""
    assert cost(location="us") == pytest.approx(0.825)
    assert cost(location="global") == pytest.approx(0.75)


def test_the_introductory_rates_end_on_new_years_eve():
    assert cost(on=datetime.date(2026, 12, 31)) == pytest.approx(0.825)
    assert cost(on=STANDARD) == pytest.approx(1.65)


def test_cached_tokens_are_priced_at_the_cache_rate_instead_of_the_input_rate():
    """1M prompt, 900K of it a cache hit: 100K × $0.825 + 900K × $0.0825 = $0.15675."""
    assert cost(cached=900_000) == pytest.approx(0.15675)


def test_thinking_is_billed_as_output():
    """The caller folds thinking into ``output``; 1M of it at $4.125 non-global."""
    assert cost(prompt=0, output=1_000_000) == pytest.approx(4.125)


def test_tool_results_fed_back_are_billed_as_input():
    assert cost(prompt=0, tool_prompt=1_000_000) == pytest.approx(0.825)


def test_the_long_context_tier_reprices_every_token_of_the_call():
    """3.1 Pro Preview, global: ≤200K is $2/$12, >200K is $4/$18 — on input *and* output."""
    short = cost("gemini-3.1-pro-preview", "global", prompt=LONG_CONTEXT_THRESHOLD, output=1000)
    long = cost("gemini-3.1-pro-preview", "global", prompt=LONG_CONTEXT_THRESHOLD + 1, output=1000)
    assert short == pytest.approx((200_000 * 2.00 + 1000 * 12.00) / 1e6)
    assert long == pytest.approx((200_001 * 4.00 + 1000 * 18.00) / 1e6)


def test_an_endpoint_class_the_page_does_not_price_is_not_priced():
    """The 3.1 Pro preview is listed at Global only, so at ``us`` it has no stated rate."""
    assert cost("gemini-3.1-pro-preview", "us") is None


def test_audio_is_priced_apart_where_the_page_prices_it_apart():
    """3.1 Flash-Lite, non-global: text $0.275/M, audio $0.55/M; 600K text + 400K audio."""
    priced = cost("gemini-3.1-flash-lite", audio_prompt=400_000)
    assert priced == pytest.approx((600_000 * 0.275 + 400_000 * 0.55) / 1e6)


def test_audio_rides_the_input_rate_where_the_row_covers_every_modality():
    assert cost(audio_prompt=400_000) == cost()


def test_a_cell_the_table_leaves_unstated_prices_nothing():
    """2.5 Flash's long-context audio cell is left unstated — so such a call has no figure."""
    long_audio = cost(
        "gemini-2.5-flash", prompt=LONG_CONTEXT_THRESHOLD + 1, audio_prompt=1000, output=10
    )
    assert long_audio is None
    assert cost("gemini-2.5-flash", prompt=LONG_CONTEXT_THRESHOLD + 1, output=10) is not None


def test_an_unknown_model_is_not_priced():
    assert cost("gemini-9-experimental") is None
    assert not known("gemini-9-experimental")


def test_a_resource_path_prices_as_its_bare_model_id():
    path = "projects/nova-project/locations/us/publishers/google/models/gemini-3.8-flash"
    assert cost(path) == cost("gemini-3.8-flash")
    assert known("publishers/google/models/gemini-3.8-flash")


@pytest.mark.parametrize(
    "traffic", ["ON_DEMAND_PRIORITY", "ON_DEMAND_FLEX", "PROVISIONED_THROUGHPUT"]
)
def test_a_non_standard_traffic_tier_is_not_priced(traffic):
    usage = Usage(prompt=1000, cached=0, output=10, traffic=traffic)
    assert call_cost("gemini-3.8-flash", "us", usage, on=INTRO) is None


def test_standard_traffic_is_priced():
    usage = Usage(prompt=1_000_000, cached=0, output=0, traffic="ON_DEMAND")
    assert call_cost("gemini-3.8-flash", "us", usage, on=INTRO) == pytest.approx(0.825)


def test_an_unnamed_tier_is_standard_only_when_the_request_asked_for_none():
    usage = Usage(prompt=1_000_000, cached=0, output=0)
    assert call_cost("gemini-3.8-flash", "us", usage, on=INTRO) == pytest.approx(0.825)
    assert call_cost("gemini-3.8-flash", "us", usage, service_tier="priority", on=INTRO) is None


@pytest.mark.parametrize("tier", ["standard", "unspecified", "STANDARD"])
def test_asking_for_the_standard_tier_by_name_is_still_standard(tier):
    usage = Usage(prompt=1_000_000, cached=0, output=0)
    assert call_cost("gemini-3.8-flash", "us", usage, service_tier=tier, on=INTRO) == pytest.approx(
        0.825
    )


def test_an_unnamed_traffic_tier_is_not_standard_when_the_request_asked_for_priority():
    usage = Usage(prompt=1000, cached=0, output=10, traffic="TRAFFIC_TYPE_UNSPECIFIED")
    assert call_cost("gemini-3.8-flash", "us", usage, service_tier="priority", on=INTRO) is None
    assert call_cost("gemini-3.8-flash", "us", usage, on=INTRO) is not None


def test_no_usage_is_no_cost():
    assert call_cost("gemini-3.8-flash", "us", None, on=INTRO) is None


def test_usage_metadata_is_read_as_vertex_reports_it():
    meta = SimpleNamespace(
        prompt_token_count=1200,
        candidates_token_count=30,
        thoughts_token_count=70,
        cached_content_token_count=1000,
        tool_use_prompt_token_count=None,
        prompt_tokens_details=[
            SimpleNamespace(modality=SimpleNamespace(value="TEXT"), token_count=1100),
            SimpleNamespace(modality=SimpleNamespace(value="AUDIO"), token_count=100),
        ],
        cache_tokens_details=None,
        traffic_type=SimpleNamespace(value="ON_DEMAND"),
    )
    assert from_usage_metadata(meta) == Usage(
        prompt=1200, cached=1000, output=100, audio_prompt=100, traffic="ON_DEMAND"
    )


def test_a_usage_block_with_no_counts_is_no_usage():
    assert from_usage_metadata(SimpleNamespace(prompt_token_count=None)) is None
    assert from_usage_metadata(None) is None


def test_the_billing_day_is_googles_pacific_day():
    late_utc = datetime.datetime(2027, 1, 1, 5, 0, tzinfo=datetime.timezone.utc)
    assert billing_day(late_utc) == datetime.date(2026, 12, 31)


def test_every_row_is_well_formed():
    """Periods in order and non-overlapping; every model priced somewhere; no negative rate."""
    for model, periods in RATES.items():
        assert model == model.lower() and model.startswith("gemini-")
        for earlier, later in itertools.pairwise(periods):
            assert earlier.end is not None and later.start == earlier.end + datetime.timedelta(1)
        for period in periods:
            tiers = [t for cls in (period.global_, period.non_global) if cls for t in cls]
            assert tiers, model
            for tier in tiers:
                for rate in (tier.input, tier.cached, tier.output, tier.audio_input):
                    assert rate is None or rate >= 0


# --- Google Search grounding (issue #656) ------------------------------------------------------


def test_every_priced_model_has_a_grounding_scheme():
    assert set(GROUNDING) == set(RATES)


def test_a_gemini_3_search_is_billed_per_grounding_query():
    """$14 per 1,000 grounding queries — the queries Google lists, whatever they found."""
    fee = grounding_cost("gemini-3.8-flash", queries=3, sourced=True)
    assert isinstance(fee, ComputedCost)
    assert fee == pytest.approx(3 * 0.014)


def test_a_gemini_2_5_search_is_billed_once_per_prompt_however_many_queries():
    """$35 per 1,000 grounding prompts — "only one charge for a Grounding Prompt"."""
    assert grounding_cost("gemini-2.5-flash", queries=5, sourced=True) == pytest.approx(0.035)
    assert grounding_cost("gemini-2.5-pro", queries=None, sourced=True) == pytest.approx(0.035)


def test_a_search_that_returned_no_sources_is_not_billed():
    assert grounding_cost("gemini-3.8-flash", queries=2, sourced=False) is None
    assert grounding_cost("gemini-2.5-flash", queries=2, sourced=False) is None


def test_a_fee_the_table_cannot_state_is_none_never_a_guess():
    assert grounding_cost("gemini-9-ultra", queries=2, sourced=True) is None
    assert (
        grounding_cost("gemini-3.8-flash", queries=None, sourced=True) is None
    )  # nothing to count


def test_grounding_tokens_are_free_on_gemini_3_and_charged_on_2_5():
    usage = Usage(prompt=100, cached=0, output=10, tool_prompt=1_000)
    on_3 = call_cost("gemini-3.8-flash", "global", usage, on=INTRO)
    assert on_3 - call_cost("gemini-3.8-flash", "global", usage, on=INTRO, grounded=True) == (
        pytest.approx(1_000 * 0.75 / 1e6)
    )
    on_25 = call_cost("gemini-2.5-flash", "global", usage, on=INTRO)
    assert call_cost("gemini-2.5-flash", "global", usage, on=INTRO, grounded=True) == on_25


def test_every_token_price_is_a_computed_cost():
    assert isinstance(cost(prompt=1_000, cached=0, output=10), ComputedCost)
