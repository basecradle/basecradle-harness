"""A free-text value cut for a log line is never cut inside a token (issue #787).

The harness scrubs a value and then truncates it; the box's Vector ``ai_scrub`` redacts the
journal again after that, before it ships. Every redaction rule with a minimum length matches a run
of ``[A-Za-z0-9_-]``, so a credential cut short of its minimum matches no rule and ships as a
prefix. The cut therefore backs off to the last character outside such a run: a token is either
whole (and redactable downstream) or absent. These tests pin that rule at the shared cut, at every
site that truncates a value bound for a log line, and pin the scrub layers' division of labor.
"""

from __future__ import annotations

import random
import re
import time
from pathlib import Path

import pytest

from basecradle_harness._exceptions import ProviderError, ProviderServerError
from basecradle_harness._observability import (
    _HEARTBEAT_RUN,
    MAX_VALUE,
    _heartbeat,
    _value,
    bounded,
    kv,
    scrub,
)
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
    characters, and the box's own pattern does the same in `re`. Matched once per space-free run, from
    the run's start, it is linear; the bound is generous so the test never flakes."""
    started = time.monotonic()
    scrub("http://" * 14285)
    scrub("http://" * 14285 + "/heartbeat/K9")
    scrub("http://" * 14285 + " x/heartbeat/K9")
    scrub("/heartbeat/" * 9090)
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
#: added to ``ai_scrub`` is added to the source's rules and to this table in the same change
#: (`docs/telemetry.md` → Credential Scrubbing).
BOX_RULE_SAMPLES = {
    "github token (gh?_)": "ghp_" + "Ab3Cd5Ef7Gh9Ij1Kl3Mn",
    "github server token": "ghs_" + "Ab3Cd5Ef7Gh9Ij1Kl3Mn",
    "github fine-grained token": "github_pat_" + "11ABCDEFG0abcdefghij",
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


def test_a_github_fine_grained_token_is_redacted_whole():
    """The box carries ``github_pat_`` too (basecradle-router#398); its run takes ``_``, so the
    redaction reaches past the underscore inside the token rather than stopping at it."""
    rendered = _value("token github_pat_11ABCDEFG0abcdefghij_klmnopqrstuv")
    assert rendered == '"token [redacted]"'


# --- a heartbeat URL, from its scheme when it has one (basecradle-router#398) ---------------------


@pytest.mark.parametrize(
    ("text", "scrubbed"),
    [
        # Userinfo before the host is a credential of its own, so a URL with a scheme goes whole.
        (
            "GET https://john:hunter2Secret@uptime.example.com/api/v1/heartbeat/Ab3Cd5Ef7Gh9 failed",
            "GET [redacted] failed",
        ),
        ("ping http://uptime.example.com/api/v1/heartbeat/Ab3Cd5Ef7Gh9", "ping [redacted]"),
        # Without one the host stays: it is not the secret, and nothing marks where the URL began.
        (
            "curl uptime.example.com/api/v1/heartbeat/Ab3Cd5Ef7Gh9",
            "curl uptime.example.com/api/v1[redacted]",
        ),
        # The box's match begins at the earliest of the two, and runs to the end of the run.
        (
            'url="https://john:pw@a.example.com/r?to=https://b.example.com/heartbeat/K9",x',
            'url="[redacted]',
        ),
        ("a/heartbeat/K9,https://john:pw@b.example.com/x", "a[redacted]"),
        # A scheme only after the last ``/heartbeat/`` starts nothing.
        ("x/heartbeat/K9?next=https://example.com", "x[redacted]"),
        # ``/heartbeat/`` needs something after it to be a URL's secret.
        (
            "https://john:pw@example.com/heartbeat/ done",
            "https://john:pw@example.com/heartbeat/ done",
        ),
        ("https:///heartbeat/K9", "[redacted]"),
    ],
)
def test_a_heartbeat_url_is_redacted_where_the_boxs_rule_begins(text, scrubbed):
    """The box's rule is ``(?:https?://\\S*)?/heartbeat/\\S+``; these are its bytes, applied linearly."""
    assert scrub(text) == scrubbed


#: The box's heartbeat rule as written in ``ai_scrub`` (basecradle-router ``4e0c50a``). Quadratic in
#: `re` on a long space-free run, so it is only ever run here, on short texts.
BOX_HEARTBEAT = re.compile(r"(?:https?://\S*)?/heartbeat/\S+")


def test_the_heartbeat_pass_redacts_exactly_the_bytes_the_boxs_rule_does():
    """A differential check over random texts built from the pieces the rule reads, so every
    interleaving of schemes, ``/heartbeat/``, separators and other characters is tried. The one
    addition to the box's bytes, a trailing bearer word, is out of these pieces by construction."""
    pieces = [
        "https://",
        "http://",
        "/heartbeat/",
        "/heartbeat",
        "heartbeat/",
        "h",
        "s",
        ":",
        "/",
        "x",
        " ",
        "\n",
        "\x85",
        "\u2028",
        "\u3000",
    ]
    rng = random.Random(806)
    for _ in range(20000):
        text = "".join(rng.choice(pieces) for _ in range(rng.randint(0, 12)))
        assert _HEARTBEAT_RUN.sub(_heartbeat, text) == BOX_HEARTBEAT.sub("[redacted]", text), text


def test_a_heartbeat_url_after_another_rules_match_is_still_taken_from_its_scheme():
    """The heartbeat pass runs first: a bearer run stops at the scheme's ``:``, and in one alternation
    it left the userinfo after it to the scheme-less arm."""
    rendered = scrub("Bearer abcdefgh.https://john:hunter2Secret@uptime.example.com/heartbeat/K9")
    assert "hunter2Secret" not in rendered and "john" not in rendered, rendered


@pytest.mark.parametrize(
    "text",
    [
        '{"url":"https://h.example.com/heartbeat/K9","auth":"Bearer eyJhbGciOiJIUzI1NiJ9.e30.SIG"}',
        "https://h.example.com/x?token=abc/heartbeat/K9&access_token=Bearer eyJhbGciOiJIUzI1NiJ9.e30",
        "uptime.example.com/heartbeat/K9;authorization=bearer\teyJhbGciOiJIUzI1NiJ9.e30",
    ],
)
def test_a_bearer_word_ending_a_heartbeat_run_still_redacts_its_token(text):
    """Taken to the run's end with the URL, the scheme word left its token in the next run with no
    rule to match it; the heartbeat pass keeps the word for the bearer rule."""
    rendered = scrub(text)
    assert "eyJhbGciOiJIUzI1NiJ9" not in rendered and "K9" not in rendered, rendered


@pytest.mark.parametrize(
    ("text", "scrubbed"),
    [
        ("x/heartbeat/K9abcBearer", "x[redacted]"),
        ("x/heartbeat/K9-bearer", "x[redacted]bearer"),
    ],
)
def test_only_a_bearer_word_not_glued_to_the_secret_is_kept(text, scrubbed):
    """A heartbeat secret that merely ends in those letters is never shortened into a kept word."""
    assert scrub(text) == scrubbed


@pytest.mark.parametrize("separator", [" ", "\t", "\n"])
def test_a_heartbeat_url_in_each_of_several_runs_is_redacted_in_each(separator):
    runs = [f"https://john:pw{n}@example.com/heartbeat/K{n}" for n in range(3)]
    assert scrub(separator.join(runs)) == separator.join(["[redacted]"] * 3)


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
    assert scrub("tokens_in=764942 cached_tokens=238277") == "tokens_in=764942 cached_tokens=238277"


# --- a vendor prefix inside a word is a word, not a credential (issue #794) ---------------------

#: One fabricated sample per **prefix-anchored** rule (`_observability._PREFIXED`) — the rules a word can contain.
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
    "abl_": "abl_" + "Kq7Zt2Wm9Rx4Lp6Vn3Hs8Dj5Fb1Gc0Ye7Ua2Ti9Ow4Qe6Nr3",
    # Shorter than a real key's 48, so no secret scanner reads the fabricated sample as one.
    "pplx-": "pplx-" + "Rm4Tq8Zw2Xk6Lp9Vb3Nc7Hd1",
    "tk_": "tk_" + "q8m2zr5wk1xt7nb4pv9hc3jd6fs0l",
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
        "tabl_partitionmaintenancewindow",
        "kabl_verbindungsueberwachungsdienst",
        "atk_objectgetaccessiblename",
        "gtk_window_set_default_size_request",  # ``tk_``'s run takes ``_``, so the word rule decides
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


# --- a refuseless key, the shape @origin read off two real ones (issue #856) ---------------------

#: ``abl_`` plus a 48-character alphanumeric body, 52 characters in all; fabricated.
REFUSELESS_KEY = PREFIX_SAMPLES["abl_"]


def test_the_refuseless_sample_has_the_vendors_shape():
    assert len(REFUSELESS_KEY) == 52 and REFUSELESS_KEY[4:].isalnum() and REFUSELESS_KEY.isascii()


@pytest.mark.parametrize(
    ("text", "scrubbed"),
    [
        # A value of its own, as a config dump or an env echo carries it.
        (REFUSELESS_KEY, "[redacted]"),
        # Free text, ended by punctuation the run does not take.
        (
            f"refuseless refused the key {REFUSELESS_KEY}, check AI_API_KEY",
            "refuseless refused the key [redacted], check AI_API_KEY",
        ),
        # Inside a vendor's quoted error body, the realistic carrier.
        (
            f'{{"error":{{"message":"Incorrect API key provided: {REFUSELESS_KEY}","code":401}}}}',
            '{"error":{"message":"Incorrect API key provided: [redacted]","code":401}}',
        ),
        (f"'{REFUSELESS_KEY}'", "'[redacted]'"),
    ],
)
def test_a_refuseless_key_is_redacted_whole(text, scrubbed):
    assert scrub(text) == scrubbed
    rendered = _value(text)
    assert REFUSELESS_KEY[4:12] not in rendered and REFUSELESS_KEY[-8:] not in rendered, rendered


@pytest.mark.parametrize(
    "token", ["tk_" + "q8m2zr5wk-xt7nb4pv9_hc3jd6fs0", "tk_" + "Q8m2zr5wk1xt7nb4pv9hc3jd6fs0l"]
)
def test_an_ntfy_token_ntfys_validator_admits_is_redacted_whole(token):
    """ntfy generates lowercase alphanumerics, and accepts ``^tk_[-_A-Za-z0-9]{29}$`` as provisioned."""
    assert len(token) == 32
    assert scrub(f"ntfy refused {token}: unauthorized") == "ntfy refused [redacted]: unauthorized"


def test_a_refuseless_key_is_redacted_on_a_log_line():
    """Through `kv`, which scrubs before it cuts, so a key past the bound is redacted, never halved."""
    line = kv(error=f"{'x ' * 100}rejected {REFUSELESS_KEY}")
    assert REFUSELESS_KEY[4:12] not in line and REFUSELESS_KEY[-8:] not in line, line


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


# --- a prefix run never swallows a literal rule's key name -----------------------------------------


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("task-runner-api-key=abc123secret", "abc123secret"),
        ("risk-model_api_key=hunter2xyz", "hunter2xyz"),
        ("sk-runner-pipeline-api-key=abc123secret", "abc123secret"),
        ("task-runner-bearer abcdefgh123", "abcdefgh123"),
        ("disk-proxy-Bearer eyJhbGciOiJIUzI1NiJ9.e30.abc", "eyJhbGciOiJIUzI1NiJ9"),
        ("AUTH_API_KEY=Bearer eyJhbGciOiJIUzI1NiJ9.e30.abc", "eyJhbGciOiJIUzI1NiJ9"),
        ("https://example.com/v1?api_key=Bearer eyJhbGciOiJIUzI1NiJ9", "eyJhbGciOiJIUzI1NiJ9"),
        ("AUTH_API_KEY=Bearer Bearer eyJhbGciOiJIUzI1NiJ9.e30.sig", "eyJhbGciOiJIUzI1NiJ9"),
        (
            "GET /v1/models?api_key=Bearer Bearer eyJhbGciOiJIUzI1NiJ9 HTTP/1.1",
            "eyJhbGciOiJIUzI1NiJ9",
        ),
        ("?token=Bearer\tBearer s3cr3tValue9", "s3cr3tValue9"),
    ],
)
def test_a_literal_rule_after_a_prefix_run_still_redacts_its_value(text, value):
    """In one alternation the leftmost match wins, and a prefix rule's run is made of the characters
    ``api-key`` and ``bearer`` are, so it consumed their key name and left the value unmatched.
    The literal rules run first."""
    rendered = scrub(f"failed: {text} rejected")
    assert value not in rendered, rendered
    assert "[redacted]" in rendered
    assert rendered.startswith("failed: ") and rendered.endswith(" rejected"), rendered


@pytest.mark.parametrize(
    ("text", "kept"),
    [
        ("task-runner-api-key=abc123secret", "task-runner-"),
        ("risk-model_api_key=hunter2xyz", "risk-model_"),
        ("task-runner-bearer abcdefgh123", "task-runner-"),
    ],
)
def test_the_word_before_a_literal_rule_keeps_its_letters(text, kept):
    assert scrub(text) == f"{kept}[redacted]"


# --- a narrow run never swallows the head of a key glued after it (issue #858) -------------------

GLUED_GITHUB = "ghp_ABCdef1234567890ABCdef1234567890abcd"


@pytest.mark.parametrize(
    ("text", "scrubbed"),
    [
        # The issue's three reproducers: each once left the second key, or part of it, in clear.
        ("hf_aaaaaaaaaaaaaaaaaaaaaa" + GLUED_GITHUB, "[redacted]"),
        (REFUSELESS_KEY + "sk-or-v1-abcdef0123456789", "[redacted]"),
        ("xhf_aaaaaaaaaaaaaaaaaaaaaa" + GLUED_GITHUB, "x[redacted]"),
        # A third key glued on goes with the second, and so does a word glued after it.
        (
            "hf_aaaaaaaaaaaaaaaaaaaaaa" + GLUED_GITHUB + "r8_Zb3Cd5Ef7Gh9Ij1Kl3Mn9 ok",
            "[redacted] ok",
        ),
        ("hf_aaaaaaaaaaaaaaaaaaaaaa" + GLUED_GITHUB + "-was-rejected ok", "[redacted] ok"),
        # An all-alphanumeric prefix glued on is a glued key too.
        (PREFIX_SAMPLES["hf_"] + "AIza" + "Ab3Cd5Ef7-Gh9Ij1Kl3Mn", "[redacted]"),
    ],
)
def test_a_key_glued_after_a_narrow_run_is_redacted_with_it(text, scrubbed):
    assert scrub(text) == scrubbed


@pytest.mark.parametrize(
    "tail",
    [
        "",
        "-sk-cdefghijkl",  # a later key after a ``-``, which no word rule ever kept
        "-hf_" + "c" * 20,
    ],
)
def test_a_match_that_took_a_glued_key_is_never_a_word(tail):
    """All lowercase and glued on the left, the chain would read as one word and keep every key in
    it; the old scrub redacted the later key on its own (the review of #858)."""
    text = "xhf_" + "a" * 20 + "ghp_" + "b" * 20 + tail
    assert scrub(text) == "x[redacted]"


@pytest.mark.parametrize("first", sorted(PREFIX_SAMPLES))
@pytest.mark.parametrize("second", sorted(PREFIX_SAMPLES))
def test_no_part_of_a_second_key_glued_after_a_first_survives(first, second):
    """Every pair of samples, glued: nothing of the second key's run is left in the line."""
    second_key = PREFIX_SAMPLES[second]
    rendered = scrub(f"key {PREFIX_SAMPLES[first]}{second_key} rejected")
    assert rendered == "key [redacted] rejected", rendered


@pytest.mark.parametrize(
    "text",
    [
        "hf_" + "Ab3Cd5Ef7Gh9Ij1Kl3Mn rejected",
        "the token hf_" + "Ab3Cd5Ef7Gh9Ij1Kl3Mn_was rejected",
        "the key hf_" + "Ab3Cd5Ef7Gh9Ij1Kl3Mn-is invalid",
    ],
)
def test_a_lone_narrow_key_matches_exactly_as_before(text):
    """No glued key, no change: the match ends where the narrow run always ended."""
    key = "hf_" + "Ab3Cd5Ef7Gh9Ij1Kl3Mn"
    assert scrub(text) == text.replace(key, "[redacted]")


def test_the_glued_key_form_is_linear():
    started = time.monotonic()
    scrub("hf_" + "a" * 200_000)
    scrub(("hf_" + "a" * 25 + "ghp") * 5_000)
    scrub("hf_" + ("a" * 21 + "_") * 8_000)
    scrub("github_pat_" + "a_" * 50_000)
    scrub("abl_" + "Ab3" * 30_000 + "sk")
    assert time.monotonic() - started < 1.0


# --- a credential sent as an ``x-api-key`` header value (issue #861) -----------------------------

#: A fabricated header value with no vendor shape, the way Wiro's key has none.
HEADER_VALUE = "wk9f8e7d6c5b4a3f2e1d0c9b8a"


@pytest.mark.parametrize(
    ("text", "scrubbed"),
    [
        (f"x-api-key: {HEADER_VALUE}", "[redacted]"),
        (
            f"X-API-Key: {HEADER_VALUE}\r\nContent-Type: application/json",
            "[redacted]\r\nContent-Type: application/json",
        ),
        (
            f'{{"x-api-key": "{HEADER_VALUE}", "x-nonce": "1731200000"}}',
            '{"[redacted]", "x-nonce": "1731200000"}',
        ),
        (f"{{'x-api-key': '{HEADER_VALUE}'}}", "{'[redacted]'}"),
        (f"[(b'x-api-key', b'{HEADER_VALUE}')]", "[(b'[redacted]')]"),
        # The unquoted assignment is the ``api_key=`` rule's, with the wider value it always took.
        (f"X-API-Key={HEADER_VALUE}&page=2", "X-[redacted]&page=2"),
        (f"x_api_key={HEADER_VALUE}", "x_[redacted]"),
        ("X-API-Key=wk9f,8e7d6;c5b4'a3f2e1", "X-[redacted]"),
        # A key of no stated shape may hold a comma or a semicolon.
        ("x-api-key: wk9f,8e7d6;c5b4 sent", "[redacted] sent"),
        # A Ruby hash, and JSON quoted inside a JSON string.
        (f'{{"x-api-key"=>"{HEADER_VALUE}"}}', '{"[redacted]"}'),
        (f"X-Api-Key => {HEADER_VALUE}", "[redacted]"),
        (f'{{\\"x-api-key\\": \\"{HEADER_VALUE}\\"}}', '{\\"[redacted]\\"}'),
        # A bearer header glued to the value goes with it, token and all.
        ("x-api-key: KBearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.sig", "[redacted]"),
        (f'"x-api-key": "{HEADER_VALUE}Bearer eyJhbGciOiJIUzI1NiJ9"', '"[redacted]"'),
        (
            f'curl -H "x-api-key: {HEADER_VALUE}" https://api.example.com',
            'curl -H "[redacted]" https://api.example.com',
        ),
        (f"x-api-key: Bearer {HEADER_VALUE}", "[redacted]"),
        (
            f"the request sent x-api-key:{HEADER_VALUE} and failed",
            "the request sent [redacted] and failed",
        ),
    ],
)
def test_an_x_api_key_header_value_is_redacted(text, scrubbed):
    assert scrub(text) == scrubbed


@pytest.mark.parametrize(
    "text",
    [
        "x-api-key",
        "the x-api-key header is missing",
        "send x-api-key, x-nonce and x-signature",
        "set the x-api-key header (see the docs)",
    ],
)
def test_a_bare_x_api_key_word_is_untouched(text):
    assert scrub(text) == text


def test_an_x_api_key_header_value_is_redacted_on_a_log_line():
    line = kv(error=f"401 Unauthorized: x-api-key: {HEADER_VALUE} rejected")
    assert HEADER_VALUE not in line
    assert line == 'error="401 Unauthorized: [redacted] rejected"'


def test_the_header_rule_is_linear():
    started = time.monotonic()
    scrub("x-api-key:" * 20_000)
    scrub("x-api-key" * 20_000)
    scrub("x-api-key: " + "a" * 200_000)
    assert time.monotonic() - started < 1.0
