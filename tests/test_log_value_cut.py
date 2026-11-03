"""A free-text value cut for a log line is never cut inside a token (issue #787).

The harness scrubs a value and then truncates it; the box's Vector ``ai_scrub`` redacts the
journal again after that, before it ships. Every redaction rule with a minimum length matches a run
of ``[A-Za-z0-9_-]``, so a credential cut short of its minimum matches no rule and ships as a
prefix. The cut therefore backs off to the last character outside such a run: a token is either
whole (and redactable downstream) or absent. These tests pin that rule at the shared cut, at every
site that truncates a value bound for a log line, and pin the scrub layers' division of labor.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from basecradle_harness._exceptions import ProviderError, ProviderServerError
from basecradle_harness._observability import _SECRETS, MAX_VALUE, _value, bounded, kv, scrub
from basecradle_harness._report import STALL_DETAIL_CAP, stall_detail

#: A credential of a shape **no** rule anywhere knows — the case the cut's own rule exists for. It is
#: fabricated, and it contains no ``z`` outside its first character, so "no ``z`` in the line" means
#: "no part of it shipped".
UNKNOWN_SHAPE = "zq_Q7mT2vXk9LpR4sWn8YbC3dFh6JgK1aEe5Uo0"
#: Padding words carrying no ``z`` and no token character run longer than four.
PAD = "word " * 150


def _no_part_or_whole(rendered: str) -> bool:
    return UNKNOWN_SHAPE in rendered or "z" not in rendered


# --- the shared cut ------------------------------------------------------------------------------


def test_a_value_that_fits_is_returned_unchanged():
    assert bounded("short", 10) == "short"
    assert bounded("x" * 10, 10) == "x" * 10


def test_a_cut_inside_a_run_backs_off_to_the_run_start():
    assert bounded("error: abcdefghij", 12) == "error:…"


def test_a_cut_that_lands_on_a_boundary_keeps_the_whole_run():
    # The head is "error: abcd" and the next character is a space: the run is whole, so it stays.
    assert bounded("error: abcd and more", 12) == "error: abcd…"


@pytest.mark.parametrize("boundary", [" ", ".", '"', "{", "/", ":"])
def test_any_character_outside_the_run_is_a_boundary(boundary):
    assert bounded(f"head{boundary}{UNKNOWN_SHAPE}", 20) == f"head{boundary}".rstrip() + "…"


def test_a_value_that_is_one_run_past_the_bound_is_absent_never_half_a_token():
    assert bounded(UNKNOWN_SHAPE, 20) == ""
    assert kv(name="t", error=UNKNOWN_SHAPE * 10) == "name=t"  # an empty value is omitted


@pytest.mark.parametrize("cap", range(1, 60))
def test_the_result_never_exceeds_the_cap(cap):
    text = f"{PAD}{UNKNOWN_SHAPE} tail"
    assert len(bounded(text, cap)) <= cap


@pytest.mark.parametrize("cap", [0, -1])
def test_a_cap_with_no_room_returns_nothing(cap):
    assert bounded("abc", cap) == ""


def test_a_marker_longer_than_the_cap_returns_nothing():
    assert bounded("alpha beta", 3, marker=" […]") == ""


def test_a_custom_marker_is_counted_inside_the_cap():
    assert bounded("alpha beta gamma", 12, marker=" […]") == "alpha […]"


# --- kv: every offset, the issue's own case ------------------------------------------------------


@pytest.mark.parametrize("offset", range(MAX_VALUE + 2))
def test_kv_never_ships_part_of_a_token_at_any_cut_position(offset):
    """The token is walked across the 240-character bound one position at a time: whichever
    character the cut lands on, the line holds the whole token or none of it."""
    rendered = _value(PAD[:offset] + UNKNOWN_SHAPE + " tail")
    assert _no_part_or_whole(rendered), rendered


def test_a_long_value_with_word_boundaries_is_still_bounded_and_quoted():
    line = kv(error="the tool exploded " * 100)
    assert len(line) < MAX_VALUE + len('error=""')
    assert line.endswith('…"')


def test_the_scrub_runs_before_the_cut_so_a_known_shape_is_redacted_whole():
    """A shape the harness knows never reaches the cut as itself: it is gone first, however close
    to the bound it sits."""
    key = "sk-proj-" + "A1b2C3d4" * 4
    for offset in range(MAX_VALUE - 40, MAX_VALUE + 2):
        rendered = _value(PAD[:offset] + key)
        assert "sk-proj" not in rendered
        assert "A1b2" not in rendered


def test_a_site_scrubs_before_it_cuts_so_no_source_rule_sees_a_cut_text():
    """The source's bearer rule needs eight characters over a wider charset than the cut's run, so a
    bearer token straddling the cut, scrubbed after it, would leave a short prefix the rule misses.
    `scrub` runs first at every log-bound site, so it is redacted whole."""
    for offset in range(150, 205):
        text = ("w " * 120)[:offset] + "Authorization: Bearer ab.cdefghijklmnopqrstuvwxyz"
        cut = bounded(scrub(text), 200)
        assert "ab." not in cut, cut


def test_the_scrub_is_linear_on_a_long_space_free_text():
    """A value is scrubbed before it is cut, so it can be as long as a response body. The heartbeat
    rule once scanned back to a URL's start for every ``http`` in a space-free run: 3 s on 100k
    characters. Anchored on its literal it is linear; the bound is generous so the test never flakes."""
    started = time.monotonic()
    scrub("http://" * 14285)
    scrub('{"url":"https://x.example.com/a.png"},' * 12000)
    assert time.monotonic() - started < 1.0


# --- the sites that cut a value bound for a log line ---------------------------------------------


def test_a_stall_detail_never_quotes_part_of_a_token():
    """`stall_detail` is the ``reason=`` of the ``wake stalled`` line and the stall note's text."""
    for offset in range(STALL_DETAIL_CAP - 50, STALL_DETAIL_CAP + 2):
        detail = stall_detail(ProviderServerError(PAD[:offset] + UNKNOWN_SHAPE, status_code=502))
        assert _no_part_or_whole(detail), detail
        assert len(detail) <= len("last error: ") + STALL_DETAIL_CAP


def test_a_stall_detail_that_would_be_empty_names_the_class():
    detail = stall_detail(ProviderServerError(UNKNOWN_SHAPE * 20, status_code=502))
    assert detail == "last error: ProviderServerError"


def test_the_vertex_inline_media_refusal_never_quotes_part_of_a_token():
    from basecradle_harness._google import GoogleProvider

    provider = GoogleProvider.__new__(GoogleProvider)
    provider._types = None  # the refusal raises before the SDK's types are touched
    media = type("Media", (), {"url": f"data:image/png;{'a.' * 20}{UNKNOWN_SHAPE},AAAA"})()
    with pytest.raises(ProviderError) as raised:
        provider._media_part(media)
    assert _no_part_or_whole(str(raised.value)), str(raised.value)


def test_the_anthropic_inline_image_refusal_never_quotes_part_of_a_token():
    from basecradle_harness._anthropic import _image_block

    image = type("Image", (), {"url": f"data:image/png;{'a.' * 20}{UNKNOWN_SHAPE},AAAA"})()
    with pytest.raises(ProviderError) as raised:
        _image_block(image)
    assert _no_part_or_whole(str(raised.value)), str(raised.value)


def test_an_unreadable_prune_watermark_never_quotes_part_of_a_token(tmp_path: Path):
    from basecradle_harness._wake import _PRUNED_THROUGH, ClaimStore

    store = ClaimStore(tmp_path)
    timeline = "019e7750-66ee-7f53-829f-13a8a710b6da"
    folder = store._folder(timeline, "messages")
    folder.mkdir(parents=True)
    (folder / _PRUNED_THROUGH).write_text("garbage " * 6 + UNKNOWN_SHAPE)
    with pytest.raises(ValueError) as raised:
        store.pruned_through(timeline, kind="messages")
    quoted = str(raised.value).split("does not hold a uuid:", 1)[1]  # past the tmp path
    assert _no_part_or_whole(quoted), quoted


# --- the scrub layers: the harness carries every shape the box does ------------------------------

#: One fabricated sample per redaction rule the box's ``ai_scrub`` carries (basecradle-router
#: ``deploy/hosts/*/vector.yaml`` and basecradle-noc ``deploy/vector.yaml``, read 2026-11-03), each
#: at the **box's own minimum length**, so a harness rule with a longer minimum fails here. The one
#: stated exception is the bearer rule (`test_a_bearer_word_in_prose_survives_the_harness_scrub`). A rule
#: added to ``ai_scrub`` is added to `_SECRETS` and to this table in the same change
#: (`docs/telemetry.md` → Credential Scrubbing).
BOX_RULE_SAMPLES = {
    "github token (gh?_)": "ghp_" + "Ab3Cd5Ef7Gh9Ij1Kl3Mn",
    "github server token": "ghs_" + "Ab3Cd5Ef7Gh9Ij1Kl3Mn",
    "bearer": "Bearer eyJhbGciOiJIUzI1NiJ9.e30.abc",
    "heartbeat url": "https://uptime.betterstack.com/api/v1/heartbeat/Ab3Cd5Ef7Gh9Ij1K",
    "basecradle user token": "bc_uat_K",
    "basecradle integration key": "bc_isk_K",
    "sk- family": "sk-ant-api03-Ab3Cd5Ef7Gh9",
    "xai": "xai-Ab3Cd5Ef7Gh9Ij1K",
    "google": "AIza" + "Ab3Cd5Ef7Gh9Ij1Kl3Mn",
    "hugging face": "hf_" + "Ab3Cd5Ef7Gh9Ij1Kl3Mn",
    "replicate": "r8_" + "Ab3Cd5Ef7Gh9Ij1Kl3Mn",
    "steel": "ste-" + "Ab3Cd5Ef7Gh9Ij1Kl3Mn",
    "api key assignment": "OPENAI_API_KEY=whatever-value",
}


@pytest.mark.parametrize("rule", sorted(BOX_RULE_SAMPLES))
def test_the_harness_redacts_every_shape_the_box_redacts(rule):
    sample = BOX_RULE_SAMPLES[rule]
    rendered = _value(f"failed with {sample} attached")
    assert "[redacted]" in rendered, rendered
    secret = sample.split("=", 1)[-1] if rule == "api key assignment" else sample[-8:]
    if rule.startswith("basecradle"):
        secret = sample  # the box's minimum is one character, so the whole token is the check
    assert secret not in rendered, rendered


def test_a_github_fine_grained_token_is_redacted_too():
    """A shape the box does not carry yet; the source carries it anyway."""
    assert "[redacted]" in _value("token github_pat_11ABCDEFG0abcdefghij_klmnopqrstuv")


def test_a_bearer_word_in_prose_survives_the_harness_scrub():
    """The one stated difference in minimum: the harness's bearer rule needs eight characters, so a
    diagnosis like "Bearer token missing" keeps its words off-fleet. No real bearer credential is
    under eight characters, so this undercuts nothing the box would have caught."""
    assert _value("Bearer token missing") == '"Bearer token missing"'


def test_an_identifier_the_harness_writes_is_not_mistaken_for_a_credential():
    for value in (
        "019e7750-66ee-7f53-829f-13a8a710b6da",
        "gen-1791030923-8abbrsSEsIk06gI224Fi",
        "max_uses_exceeded",
        "config:conflicting_routing",
    ):
        assert _value(value) == value
    assert not _SECRETS.search("tokens_in=764942 cached_tokens=238277")


# --- a vendor prefix inside a word is a word, not a credential (issue #794) ---------------------

#: One fabricated sample per **prefix-anchored** rule in `_SECRETS` — the rules a word can contain.
PREFIX_SAMPLES = {
    "sk-": "sk-ant-api03-Ab3Cd5Ef7Gh9",
    "sk-or-": "sk-or-v1-0f3a9c2b7d4e6f8a",
    "xai-": "xai-Ab3Cd5Ef7Gh9Ij1K",
    "bc_???_": "bc_tok_Ab3Cd5Ef7Gh9",
    "bc_uat_": "bc_uat_K",
    "bc_isk_": "bc_isk_K",
    "gh?_": "ghp_" + "Ab3Cd5Ef7Gh9Ij1Kl3Mn",
    "github_pat_": "github_pat_11ABCDEFG0abcdefghij_klmnopqrstuv",
    "AIza": "AIza" + "Ab3Cd5Ef7Gh9Ij1Kl3Mn",
    "hf_": "hf_" + "Ab3Cd5Ef7Gh9Ij1Kl3Mn",
    "r8_": "r8_" + "Ab3Cd5Ef7Gh9Ij1Kl3Mn",
    "ste-": "ste-" + "Ab3Cd5Ef7Gh9Ij1Kl3Mn",
}

BACKSLASH = chr(92)

#: Every left neighbor a credential can follow. The first group is a separator; the second ends in
#: a letter or digit — a serialization, a color code, a script written without spaces, a word
#: glued on by a bug — which is why no left boundary can be the test (`_is_word`).
LEFT_NEIGHBORS = {
    "start of text": "",
    "space": "key ",
    "equals": "key=",
    "colon": "key:",
    "double quote": '"',
    "underscore": "OPENAI_KEY_",
    "hyphen": "x-",
    "slash": "/v1/keys/",
    "percent-encoded space": "Bearer%20",
    "double percent-encoding": "key%253D",
    "quoted-printable": "key=3D",
    "escaped newline": "body" + BACKSLASH + "n",
    "hex escape": BACKSLASH + "x22",
    "unicode escape": BACKSLASH + "u0022",
    "astral escape": BACKSLASH + "U0001f600",
    "ANSI color": "invalid key \x1b[33m",
    "Chinese": "API\u5bc6\u94a5",
    "Japanese": "API\u30ad\u30fc",
    "Bearer with no space": "Authorization:Bearer",
    "timestamp": "2026-11-03T10:00:00Z",
    "letter": "a",
    "accented letter": "\u00e9",
    "combining mark": "e\u0301",
}


@pytest.mark.parametrize(
    "text",
    [
        "task-specification failed",
        "disk-quota-exceeded-for-org",
        "risk-assessment-pending-review",
        "the task-specification-for-this-run is invalid",
        "laughs_happily_ever_after_and_after",
        "abc_uat_unknown",
        "R\u00edsk-assessment-pending-review",  # a localized vendor sentence keeps its words too
        "G\u00e4ste-registrierungssystem-voll",
    ],
)
def test_a_vendor_prefix_inside_a_word_is_not_a_credential(text):
    assert scrub(text) == text
    assert _value(text) == (f'"{text}"' if " " in text else text)


@pytest.mark.parametrize("rule", sorted(PREFIX_SAMPLES))
@pytest.mark.parametrize("left", sorted(LEFT_NEIGHBORS))
def test_a_credential_after_every_left_neighbor_is_still_redacted(rule, left):
    """A key glued to a letter or digit is still a key: its run holds a digit or a capital."""
    prefix = LEFT_NEIGHBORS[left]
    rendered = scrub(f"{prefix}{PREFIX_SAMPLES[rule]} rejected")
    assert rendered == f"{prefix}[redacted] rejected", rendered


@pytest.mark.parametrize("glue", ["a", "7", "\u00e9", "\u5bc6"])
def test_a_glued_run_with_a_digit_or_a_capital_is_redacted_however_wordlike(glue):
    """The stated residual, in the safe direction: the scrub errs toward redaction."""
    assert scrub(f"{glue}sk-specification2") == f"{glue}[redacted]"
    assert scrub(f"{glue}sk-Specification") == f"{glue}[redacted]"


def test_a_prefix_that_holds_a_digit_is_not_read_as_the_run():
    """``r8_``'s own digit is the prefix, never evidence about the run after it."""
    assert scrub("xr8_" + "abcdefghijklmnopqrstuvwx") == "xr8_" + "abcdefghijklmnopqrstuvwx"
    assert scrub("r8_" + "abcdefghijklmnopqrstuvwx") == "[redacted]"


def test_an_unglued_lowercase_run_is_still_redacted():
    """Only a glued match can be a word: at a token start every prefixed run is a credential."""
    assert scrub("key sk-abcdefghij") == "key [redacted]"
    assert scrub("x-hf_" + "abcdefghijklmnopqrstuvwx") == "x-[redacted]"


@pytest.mark.parametrize(
    "secret",
    [
        "api_key=hunterpass",
        "bearer abcdefghijk",
        "/heartbeat/abcdefghij",
        "?token=abcdefgh",
    ],
)
@pytest.mark.parametrize("glue", ["x", "7", "密"])
def test_a_non_prefix_rule_is_redacted_whatever_precedes_it(secret, glue):
    """Only a vendor prefix can be part of a word: every other rule redacts wherever it matches,
    lowercase value and glued neighbor included."""
    assert "[redacted]" in scrub(f"config {glue}{secret}")
    assert secret.split("=")[-1].split("/")[-1] not in scrub(f"config {glue}{secret}")


@pytest.mark.parametrize("glue", ["7", "é", "密"])
def test_a_word_glued_by_a_digit_a_combining_mark_or_any_script_is_kept(glue):
    assert scrub(f"{glue}sk-specification") == f"{glue}sk-specification"
