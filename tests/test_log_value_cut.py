"""A free-text value cut for a log line is never cut inside a token (issue #787).

The harness scrubs a value and then truncates it; the box's Vector ``ai_scrub`` redacts the
journal again after that, before it ships. Every redaction rule with a minimum length matches a run
of ``[A-Za-z0-9_-]``, so a credential cut short of its minimum matches no rule and ships as a
prefix. The cut therefore backs off to the last character outside such a run: a token is either
whole (and redactable downstream) or absent. These tests pin that rule at the shared cut, at every
site that truncates a value bound for a log line, and pin the scrub layers' division of labor.
"""

from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path

import pytest

from basecradle_harness import _observability
from basecradle_harness._exceptions import ProviderError, ProviderServerError
from basecradle_harness._observability import (
    _HEARTBEAT_RUN,
    MAX_VALUE,
    REDACTED_VALUE,
    _heartbeat,
    _value,
    bounded,
    kv,
    redact_credentials,
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
    # A credential-named assignment keeps its name and loses its value to the router's marker
    # (issue #918); every other shape becomes `[redacted]`.
    marker = REDACTED_VALUE if rule == "api key assignment" else "[redacted]"
    assert marker in rendered, rendered
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
    assert "[redacted]" in rendered or REDACTED_VALUE in rendered  # a credential name's: #918
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


# --- an assignment's quoted value, and a bearer header glued to a value (review of #857) ---------


@pytest.mark.parametrize(
    ("text", "scrubbed"),
    [
        # A credential-named assignment keeps its name, and its quoted value goes whole (#918).
        (
            'config API_KEY="wk9f8e7d6c5b4a3f2e1d0c9b8a" set',
            "config API_KEY='<a redacted value>' set",
        ),
        ("api_key='wk9f8e7d6c5b'", "[redacted]"),
        ("api_key=KBearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0 sent", "[redacted] sent"),
        ("GET /v1?token=KBearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0&x=1", "GET /v1[redacted]&x=1"),
        ("GET /v1?api_key=wk9f8e7d6c5b&x=1", "GET /v1[redacted]&x=1"),
        ("OPENAI_API_KEY=Bearer eyJhbGciOiJIUzI1NiJ9", "OPENAI_API_KEY='<a redacted value>'"),
    ],
)
def test_an_assignments_value_is_redacted_however_it_opens_and_whatever_is_glued_to_it(
    text, scrubbed
):
    assert scrub(text) == scrubbed


def test_the_assignment_rules_are_linear():
    started = time.monotonic()
    scrub("api_key=" + "a" * 200_000)
    scrub("api_key=" + ("a" * 5 + "Bearer ") * 20_000)
    scrub("?token=" + "bearer" * 30_000)
    scrub('API_KEY="' * 20_000)
    assert time.monotonic() - started < 1.0


# --- an `api_key=` value in escaped quotes, and a value ending in the next label (issue #934) -----

#: A key of no vendor's shape, fabricated: only the label it follows can say what it is.
SHAPELESS_KEY = "K9fabricatedSecret07"

#: A quote escaped as a third JSON string nested in two others writes it: seven backslashes.
ESCAPED_3 = "\\" * 7 + '"'

#: A fabricated bearer token, the shape the issue's lines carried.
FAKE_JWT = "eyJfabricatedSecretValue01"


@pytest.mark.parametrize(
    ("text", "scrubbed"),
    [
        # The issue's four lines, found by the router and the NOC at 0.198.2.
        (f'api_key=\\"{SHAPELESS_KEY}\\" n', '[redacted]\\" n'),
        (f'{{"h":"api_key=\\"{SHAPELESS_KEY}\\""}}', '{"h":"[redacted]\\""}'),
        (f'api_key=x,api_key="{FAKE_JWT}"', '[redacted]"'),
        (f'api-key="Bearer api_key="{FAKE_JWT}', "[redacted]"),
        # An escaped quote escaped again, as JSON inside a JSON string writes it.
        (f'api_key=\\\\\\"{SHAPELESS_KEY}\\\\\\" n', '[redacted]\\\\\\" n'),
        # A label ending where the value stops, before the backslashes that escape the stop.
        (f'api_key=x,api_key=\\"{SHAPELESS_KEY}\\" n', '[redacted]\\" n'),
        # Escaped deeper than the label is long, three JSON strings in: the backslashes are skipped.
        (f"api_key=x,api_key={ESCAPED_3}{SHAPELESS_KEY}{ESCAPED_3} n", f"[redacted]{ESCAPED_3} n"),
        # A label before one scheme word the value stops after.
        ("api_key=x,api_key=Bearer abc n", "[redacted] n"),
        ("?token=x?token=Bearer abc n", "[redacted] n"),
        # A label just before the glued scheme word a value ended in, with its wider run after it.
        ('api_key=\\"api_key=bearer\tX-API-KEY= n', "[redacted] n"),
        # A label inside a value in escaped quotes reads to the same stop and leaves the same escape
        # (issue #937: until then it took the escape, and the line no longer parsed).
        (f'api_key=\\"api-key={SHAPELESS_KEY}\\" n', '[redacted]\\" n'),
        # The header's read-on is the same piece (the review of #929).
        (f'x-api-key: "x-api-key": "{SHAPELESS_KEY}"', '[redacted]"'),
    ],
)
def test_a_literal_value_is_read_whole_however_it_is_escaped_and_whatever_label_it_ends_in(
    text, scrubbed
):
    assert scrub(text) == scrubbed
    assert scrub(scrubbed) == scrubbed


@pytest.mark.parametrize("depth", range(4))
def test_an_api_key_in_a_json_string_leaves_the_line_parseable(depth):
    line = json.dumps({"h": f'api_key="{SHAPELESS_KEY}" sent'})
    for _ in range(depth):
        line = json.dumps({"log": line})
    scrubbed = scrub(line)
    assert SHAPELESS_KEY not in scrubbed
    for _ in range(depth + 1):
        scrubbed = json.loads(scrubbed)
        scrubbed = scrubbed.get("log", scrubbed.get("h"))
    assert scrubbed == '[redacted]" sent'


@pytest.mark.parametrize("value", ["\\", "\\\\", "x\\", "K9\\\\"])
@pytest.mark.parametrize("separators", [(",", ":"), (", ", ": ")])
def test_a_quote_after_an_even_run_of_backslashes_is_no_opener(value, separators):
    # The review of #934: an even run escapes itself, so the quote after it closes the JSON string,
    # and taken as an escaped opener it took that close and the line no longer parsed.
    line = json.dumps({"h": f"api_key={value}", "k": "v"}, separators=separators)
    assert json.loads(scrub(line)) == {"h": "[redacted]", "k": "v"}


def test_the_odd_run_before_the_close_stays_whole():
    # The value ends in its own escaped backslash, and the run before the close is left whole
    # (issue #937): which of its backslashes are the close's depends on a depth the run cannot tell.
    assert scrub('api_key=\\"K9fabricated\\\\\\" n') == '[redacted]\\\\\\" n'
    line = json.dumps({"h": 'api_key="K9fabricated\\" sent'})
    assert json.loads(scrub(line)) == {"h": '[redacted]\\" sent'}


def test_no_chain_of_labels_or_escapes_lets_an_api_key_through():
    rng = random.Random(934)
    labels = ["api_key=", "api-key=", "APIKEY=", "X-API-KEY=", "?token=", "x-api-key: "]
    pieces = [
        *labels, "x", "abc", ",", '"', '\\"', '\\\\\\"', "Bearer ", "bearer\t", "KBearer", "&",
    ]  # fmt: skip
    for _ in range(20_000):
        head = "".join(rng.choice(pieces) for _ in range(rng.randint(0, 8)))
        opener = rng.choice(["", '"', '\\"', '\\\\\\"', "Bearer ", '"Bearer '])
        tail = rng.choice(["", " n", '\\"'])
        text = f"{head}{rng.choice(labels[:4])}{opener}{SHAPELESS_KEY}{tail}"
        assert SHAPELESS_KEY not in scrub(text), repr(text)


def test_the_read_on_is_linear():
    started = time.monotonic()
    scrub('api_key=x,api_key="' * 20_000)
    scrub('api_key=\\"' * 20_000)
    scrub("api_key=" + "\\" * 200_000 + '"')
    scrub('api_key=\\"' + "api_key=bearer\t" * 20_000)
    scrub('api_key=\\"' + "a," * 100_000 + '\\"')
    scrub('api_key=\\"' + "api_key=" * 20_000 + '\\"')
    scrub("?token=x" + "?token=Bearer " * 20_000)
    scrub("api_key=x" + ("api_key=bearer abcdefghij=" * 10_000))
    assert time.monotonic() - started < 2.0


# --- an `x-api-key` header escaped more than once (issue #935) ----------------------------------

#: A quote escaped as JSON inside two nested JSON strings writes it: three backslashes.
ESCAPED_2 = "\\" * 3 + '"'


@pytest.mark.parametrize("depth", range(4))
def test_an_x_api_key_header_is_redacted_at_any_json_depth(depth):
    # Until 0.198.3 the label and the opener took a quote escaped once, so depths 2 and 3 shipped.
    line = json.dumps({"x-api-key": SHAPELESS_KEY})
    for _ in range(depth):
        line = json.dumps({"log": line})
    scrubbed = scrub(line)
    assert SHAPELESS_KEY not in scrubbed
    assert scrub(scrubbed) == scrubbed
    for _ in range(depth):
        scrubbed = json.loads(scrubbed)["log"]  # every enclosing JSON string still parses
    assert scrubbed == '{"[redacted]"}'


@pytest.mark.parametrize(
    ("text", "scrubbed"),
    [
        # The issue's header dump, its value in quotes escaped twice over.
        (f"x-api-key: {ESCAPED_2}{SHAPELESS_KEY}{ESCAPED_2} n", f"[redacted]{ESCAPED_2} n"),
        (f"x-api-key{ESCAPED_2}=>{ESCAPED_2}{SHAPELESS_KEY}{ESCAPED_2}", f"[redacted]{ESCAPED_2}"),
        (f"x-api-key{ESCAPED_3}, {ESCAPED_3}{SHAPELESS_KEY}{ESCAPED_3}", f"[redacted]{ESCAPED_3}"),
        (f"x-api-key: b{ESCAPED_2}{SHAPELESS_KEY}", "[redacted]"),
        # A value ending where the next header, escaped deeper, begins is read on through it.
        (
            f"x-api-key: x,x-api-key{ESCAPED_2}: {ESCAPED_2}{SHAPELESS_KEY}{ESCAPED_2} n",
            f"[redacted]{ESCAPED_2} n",
        ),
        # The differential's stated case: the header's span now starts before the credential name
        # inside it, so it is written `[redacted]`, never the marker whose re-reading took `tail`,
        # which follows a real quote and is no rule's value.
        (
            f"x-api-key\\\\\\':A_TOKEN=x-api-key\\' :{SHAPELESS_KEY}x-api-key\\\\\",tail",
            '[redacted]\\\\",tail',
        ),
    ],
)
def test_an_x_api_key_header_escaped_deeper_than_once_is_redacted(text, scrubbed):
    assert scrub(text) == scrubbed
    assert scrub(scrubbed) == scrubbed


@pytest.mark.parametrize("value", ["\\", "\\\\", "x\\", "K9\\\\"])
@pytest.mark.parametrize("separators", [(",", ":"), (", ", ": ")])
def test_a_header_quote_after_an_even_run_of_backslashes_is_no_opener(value, separators):
    # As for `api_key=` (#934): the quote after an even run closes the JSON string, and taken as an
    # escaped opener the value ran on past it, and the line no longer parsed.
    line = json.dumps({"h": f"x-api-key: {value}", "k": "v"}, separators=separators)
    assert json.loads(scrub(line))["k"] == "v"


@pytest.mark.parametrize("value", ["\\", "\\\\", "x\\", "K9\\\\"])
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_a_nested_even_run_leaves_the_outermost_line_parseable(value, depth):
    # The stated limit (the review of #935): an encoding pass turns an inner string's even run into
    # an odd one, so that inner string can lose its close. The line itself always parses. (An
    # unquoted `api_key=` value nested so keeps every string whole since #937.)
    line = json.dumps({"h": f"x-api-key: {value}", "k": "v"}, separators=(",", ":"))
    for _ in range(depth):
        line = json.dumps({"log": line})
    assert set(json.loads(scrub(line))) == {"log"}


def test_no_header_escaped_at_any_depth_lets_its_key_through():
    rng = random.Random(935)
    escapes = ["", "\\", "\\" * 3, "\\" * 7]
    pieces = [
        "x-api-key: ", "x-api-key", "api_key=", "x", "abc", ",", '"', '\\"', ESCAPED_2, ESCAPED_3,
        "Bearer ", "bearer\t", "KBearer", "&", ": ", "=>",
    ]  # fmt: skip
    for _ in range(20_000):
        head = "".join(rng.choice(pieces) for _ in range(rng.randint(0, 6)))
        quote = rng.choice("\"'")
        label = rng.choice(
            [
                "x-api-key: ",
                "X_API_KEY=> ",
                f"x-api-key{rng.choice(escapes)}{quote}{rng.choice([':', ': ', '=>', ',', ' ='])}",
            ]
        )
        opener = rng.choice(["", " ", "b"]) + rng.choice(escapes) + rng.choice(["", quote])
        if opener.endswith("\\"):
            opener = ""  # a backslash with no quote after it opens nothing
        scheme = rng.choice(["", "Bearer ", "bearer\t"])
        tail = rng.choice(["", " n", rng.choice(escapes) + quote])
        text = f"{head}{label}{opener}{scheme}{SHAPELESS_KEY}{tail}"
        assert SHAPELESS_KEY not in scrub(text), repr(text)


def test_the_escaped_header_rule_is_linear():
    started = time.monotonic()
    scrub("x-api-key" + "\\" * 200_000 + '":')
    scrub("x-api-key" + "\\" * 200_001 + "x")
    scrub("x-api-key: " + "\\" * 200_000 + '"')
    scrub(f"x-api-key{ESCAPED_2}: " * 20_000)
    scrub(f"x-api-key: x,x-api-key{ESCAPED_2}: " * 10_000)
    bounded(f"x-api-key{ESCAPED_2}: {ESCAPED_2}" * 10_000, 100_000)
    assert time.monotonic() - started < 2.0


# --- an `api_key=` value leaves the odd run before the quote it stops at (issue #937) -----------

#: Values ending in backslashes after something else, and one holding a key.
BACKSLASH_VALUES = ["x\\", "K9\\\\", f"{SHAPELESS_KEY}\\", f"{SHAPELESS_KEY}\\\\\\"]


def _levels(line: str) -> list[dict]:
    """Every JSON object `line` holds, outermost first, each nested in the one before as ``log``."""
    objects = [json.loads(line)]
    while isinstance(objects[-1].get("log"), str):
        objects.append(json.loads(objects[-1]["log"]))
    return objects


def test_the_issue_line_keeps_its_closing_quote():
    # Until 0.198.3 the run took the backslash escaping the line's own closing quote.
    line = json.dumps({"log": json.dumps({"h": "api_key=x\\", "k": "v"}, separators=(",", ":"))})
    assert scrub(line) == '{"log": "{\\"h\\":\\"[redacted]\\\\\\\\\\",\\"k\\":\\"v\\"}"}'


@pytest.mark.parametrize("value", BACKSLASH_VALUES)
@pytest.mark.parametrize("depth", [0, 1, 2, 3])
@pytest.mark.parametrize("separators", [(",", ":"), (", ", ": ")])
def test_a_value_ending_in_backslashes_keeps_every_string_it_is_nested_in(value, depth, separators):
    # The run is left whole when it is odd, so every enclosing string parses, not only the line's,
    # and the value's own backslashes stay with it. At depth 0 the run before the close is even, the
    # value's own, and goes with the value.
    line = json.dumps({"h": f"api_key={value}", "k": "v"}, separators=separators)
    for _ in range(depth):
        line = json.dumps({"log": line})
    scrubbed = scrub(line)
    assert SHAPELESS_KEY not in scrubbed
    assert scrub(scrubbed) == scrubbed
    *_, inner = _levels(scrubbed)
    assert inner == {
        "h": "[redacted]" + ("\\" * (len(value) - len(value.rstrip("\\"))) if depth else ""),
        "k": "v",
    }


@pytest.mark.parametrize("value", ["\\", "\\\\"])
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_a_nested_value_of_backslashes_alone_leaves_the_line_parseable(value, depth):
    # The stated limit (#934, #935): nested, a value that is backslashes alone reaches the line as an
    # odd run before its real close, which is read as an escaped opener, so that inner string loses
    # its close. The line itself parses.
    line = json.dumps({"h": f"api_key={value}", "k": "v"}, separators=(",", ":"))
    for _ in range(depth):
        line = json.dumps({"log": line})
    assert set(json.loads(scrub(line))) == {"log"}


@pytest.mark.parametrize("value", ["\\", "K9\\"])
@pytest.mark.parametrize("separators", [(",", ":"), (", ", ": ")])
def test_an_escaped_opener_before_a_real_close_leaves_no_backslash_behind(value, separators):
    # #934's escaped opener left as many backslashes as it carried, even before a real quote, so the
    # one it left escaped the string's close and the line no longer parsed.
    line = json.dumps({"h": f'api_key="{value}', "k": "v"}, separators=separators)
    assert json.loads(scrub(line)) == {"h": "[redacted]", "k": "v"}


def test_no_api_key_value_breaks_the_line_or_a_string_nested_in_it():
    # No key ships and the line parses. Where nothing before the value can be read as an escaped
    # opener (no quote, no backslash), every JSON string the line held parses too. Bearer scheme
    # words and credential names are left out: their rules read a shell word, which may run past a
    # quote, and the scrub errs toward redaction there (`_past_marker`). So is ``&``: ``&api_key=``
    # is a query label, whose run takes quotes and backslashes alike. So is an empty value, which
    # opens at the string's own close.
    rng = random.Random(937)
    clean = ["api_key=", "api-key=", "x", "abc", ",", " "]
    pieces = [*clean, '"', "\\", "\\\\", '\\"']
    for _ in range(20_000):
        whole = rng.random() < 0.5
        head = "".join(rng.choice(clean if whole else pieces) for _ in range(rng.randint(0, 5)))
        opener = "" if whole else rng.choice(["", '"'])
        key = rng.choice([SHAPELESS_KEY, "x"])
        tail = rng.choice(["", " n", '"', "\\" * rng.randint(1, 4)])
        line = json.dumps(
            {"h": f"{head}api_key={opener}{key}{tail}", "k": "v"},
            separators=rng.choice([(",", ":"), (", ", ": ")]),
        )
        for _ in range(rng.randint(0, 3)):
            line = json.dumps({"log": line})
        scrubbed = scrub(line)
        assert SHAPELESS_KEY not in scrubbed, repr(line)
        assert set(json.loads(scrubbed)) == set(json.loads(line)), repr(line)
        if whole:
            assert _levels(scrubbed)[-1]["k"] == "v", repr(line)


def test_the_odd_run_is_read_in_linear_time():
    started = time.monotonic()
    scrub("api_key=x" + "\\" * 200_001 + '"')
    scrub("api_key=x" + "\\" * 200_000 + '"')
    scrub('api_key=x\\",' * 20_000)
    scrub('api_key=x,api_key=\\\\\\"' * 20_000)
    assert time.monotonic() - started < 2.0


# --- no value opens at the close of the JSON string it sits in (issue #939) ---------------------

#: A literal label of each rule whose value may open with a ``"``, each ending a word.
CLOSE_LABELS = ["api_key=", "Api-Key=", "x-api-key: "]


@pytest.mark.parametrize(
    ("text", "scrubbed"),
    [
        # The issue's empty value, whose opener was the string's own close.
        ('{"h":"api_key=","k":"v"}', '{"h":"api_key=","k":"v"}'),
        ('{"h": "see api_key=", "k": "v"}', '{"h": "see api_key=", "k": "v"}'),
        # The review's: a read-on reaching an inner label whose opener is the string's close, and a
        # header whose opener is.
        ('{"msg": "x,Api-Key=apikey=", "k": "v"}', '{"msg": "x,[redacted]", "k": "v"}'),
        ('{"h": "x-api-key\\\\\\":"}', '{"h": "x-api-key\\\\\\":"}'),
        ('{"h": "x-api-key: ", "k": "v"}', '{"h": "x-api-key: ", "k": "v"}'),
        # A value that is structure and more is a value. A `'` closes no JSON string, so it opens:
        # dropped, the header's span no longer reaches the marker's re-reading, and the key shipped.
        (f'api_key=",{SHAPELESS_KEY}" n', '[redacted]" n'),
        (
            f'OPENAI_API_KEY=Bearer ="x-api-key": \',\\\\{SHAPELESS_KEY} n',
            f"OPENAI_API_KEY={REDACTED_VALUE} n",
        ),
    ],
)
def test_no_value_opens_at_the_close_of_the_string_it_sits_in(text, scrubbed):
    assert scrub(text) == scrubbed
    assert scrub(scrubbed) == scrubbed


@pytest.mark.parametrize("label", CLOSE_LABELS)
@pytest.mark.parametrize("value", [SHAPELESS_KEY, f'"{SHAPELESS_KEY}"', ""])
@pytest.mark.parametrize("depth", range(4))
@pytest.mark.parametrize("separators", [(",", ":"), (", ", ": ")])
def test_a_literal_value_keeps_every_string_it_is_nested_in(label, value, depth, separators):
    line = json.dumps({"h": f"url {label}{value}", "k": "v"}, separators=separators)
    for _ in range(depth):
        line = json.dumps({"log": line}, separators=separators)
    scrubbed = scrub(line)
    assert SHAPELESS_KEY not in scrubbed
    assert scrub(scrubbed) == scrubbed
    levels = _levels(scrubbed)
    assert len(levels) == depth + 1  # the line and every string nested in it parse
    assert levels[-1] == {
        "h": "url [redacted]" + ('"' if value.startswith('"') else "") if value else f"url {label}",
        "k": "v",
    }


def test_a_credential_named_assignment_reads_a_shell_word_across_the_close():
    # A stated limit (#939): the value is read as the router reads it, a shell word, which spans
    # quotes. Inside a JSON string the two readings of that quote cannot both be right, and the one
    # that takes more is the safe one, so the line loses its close and no key ships.
    line = json.dumps({"h": 'A_TOKEN=x,api_key="K9fabricated"', "k": "v"}, separators=(",", ":"))
    assert scrub(line) == redact_credentials(line) == '{"h":"A_TOKEN=' + REDACTED_VALUE


def test_an_opener_before_a_value_that_is_not_a_string_still_opens():
    # A stated limit (the review of #939): the value is read to where its run stops, so structure
    # running on into a number, a literal, an object or an array is more than structure, and the
    # opener still opens, as on 0.198.3. Nested, the line parses and the inner string loses its close.
    assert scrub('["api_key=",1]') == '["[redacted]'
    assert scrub('{"api_key=":1}') == '{"[redacted]'
    nested = json.dumps({"log": json.dumps(["api_key=\\", 2], separators=(",", ":"))})
    assert set(json.loads(scrub(nested))) == {"log"}


def test_a_key_reached_only_through_a_dropped_opener_is_kept():
    # What dropping an opener costs (#939), stated in `docs/telemetry.md`: 0.198.3 read the marker
    # on through the header's opener, a close by `_closes`, to the text after it. No other kept key
    # was found, by two adversarial differentials.
    text = f'A_TOKEN=x-api-key: "]\\{SHAPELESS_KEY}'
    assert scrub(text) == f'A_TOKEN={REDACTED_VALUE} "]\\{SHAPELESS_KEY}'


def test_a_key_in_a_query_still_takes_the_close():
    # A stated limit (#939): the query value's run takes a `"`. Stopping it at one gave up keys the
    # run took (a value in doubled CSV quotes, `?token=x"K9…`, the next JSON member), and no reader
    # in the fleet parses JSON inside a message, so the field is the price and the key is not.
    assert scrub('{"h":"url?api_key=K9fabricated","k":"v"}') == '{"h":"url[redacted]'
    csv = f'GET,"https://h.example.com/v1?token=""{SHAPELESS_KEY}""",200'
    assert scrub(csv) == 'GET,"https://h.example.com/v1[redacted]'


def test_dropping_an_opener_at_a_close_keeps_no_key_and_breaks_no_line(monkeypatch):
    # A differential against the reading before #939, where every opener opened. Credential names
    # and bearer scheme words are left out: their rules read a shell word or an escaped token, which
    # may run past a quote (the limits stated above), and a key 0.198.3 reached only by reading its
    # marker on past such an opener is stated in `docs/telemetry.md`.
    rng = random.Random(939)
    pieces = [
        *CLOSE_LABELS, "url", "x", "abc", ",", " ", "&", ":", "}", "]", '"', "'", '\\"', ESCAPED_2,
        "\\", "\\\\", '""',
    ]  # fmt: skip
    cases = []
    for _ in range(10_000):
        parts = [rng.choice(pieces) for _ in range(rng.randint(1, 7))]
        parts.insert(rng.randint(0, len(parts)), rng.choice(CLOSE_LABELS))
        parts.insert(rng.randint(1, len(parts)), SHAPELESS_KEY)
        line = json.dumps(
            {"h": "".join(parts), "k": "v"}, separators=rng.choice([(",", ":"), (", ", ": ")])
        )
        for _ in range(rng.randint(0, 3)):
            line = json.dumps({"log": line})
        cases.append((line, scrub(line)))
    monkeypatch.setattr(_observability, "_closes", lambda match: False)
    for line, new in cases:
        old = _observability.scrub(line)
        assert SHAPELESS_KEY in old or SHAPELESS_KEY not in new, repr(line)
        assert _depth_parsed(new) >= _depth_parsed(old), repr(line)


def _depth_parsed(line: str) -> int:
    """How many levels of `line` parse, outermost first: 0 when the line itself does not."""
    depth = 0
    while True:
        try:
            obj = json.loads(line)
        except ValueError:
            return depth
        depth += 1
        if not isinstance(obj, dict) or not isinstance(obj.get("log"), str):
            return depth
        line = obj["log"]


def test_an_opener_at_a_close_is_skipped_in_linear_time():
    started = time.monotonic()
    scrub('api_key=","' * 20_000)
    scrub('x-api-key: ","' * 20_000)
    scrub('api_key=\\",\\"' * 20_000)
    scrub("api_key=" + '",' * 100_000)
    assert time.monotonic() - started < 2.0


# --- the cut treats a literal label as one token (issue #931) -----------------------------------

#: Every literal rule's label, in each spelling its rule allows: a query parameter, an ``api_key=``
#: assignment, an ``x-api-key`` header with each separator, and a heartbeat URL.
LABELS = [
    *(
        f"{lead}{name}="
        for lead in "?&"
        for name in ("api-key", "api_key", "apikey", "Access-Token", "access_token", "accesstoken")
        + ("token", "KEY")
    ),
    "api-key=",
    "api_key=",
    "APIKEY=",
    *(
        name + separator
        for name in ("x-api-key", "X_API_KEY", "x-apikey")
        for separator in (":", " :", "=>", " =>", '":', "'=", '\\":', "',", '\\\\\\":')
    ),
    "https://h.example.com/heartbeat/",
    "h/heartbeat/",
]


@pytest.mark.parametrize(
    ("text", "cap", "cut"),
    [
        # The review of #929: each label cut to its `…`, which the formatter's second scrub read as
        # its value and wrote as `[redacted]`, past the cap of the first and without the `…` in all.
        ("see ?token= tail", 13, "see…"),
        ("see api_key= tail", 14, "see…"),
        ("see x-api-key: tail", 16, "see…"),
        # A scheme word's token and an `x-api-key` value are read across blanks, so no blank
        # between a label and the marker would keep the marker out of the value.
        ("see x-api-key: Bearer tail", 22, "see…"),
        ("see ?token=Bearer  tail", 20, "see…"),
        # A heartbeat URL whose secret the cut hides: the box's rule reads the `…` as the secret.
        ("curl h.example.com/heartbeat/ then more", 33, "curl…"),
        # A label whose own `[redacted]` the cut keeps, the `…` glued to it.
        ("see x-api-key: [redacted] more", 28, "see…"),
    ],
)
def test_a_bound_cuts_a_label_whole_when_the_marker_would_be_its_value(text, cap, cut):
    assert bounded(text, cap) == cut
    assert scrub(cut) == cut


def test_a_bound_keeps_a_label_whose_value_the_cut_does_not_reach():
    # The blank after `?token=` ends its value before the cut, so the scrub reads none there.
    assert bounded("see ?token= then more", 17) == "see ?token= then…"
    assert scrub("see ?token= then…") == "see ?token= then…"


@pytest.mark.parametrize("marker", ["…", " […]"])
def test_no_literal_label_lets_a_second_scrub_grow_a_bound_or_take_its_marker(marker):
    # Measured on `main` at `5bce26a`, with the `…`: 51 of these 30,000 texts grew past their cap
    # and 1,029 more lost the `…`.
    rng = random.Random(931)
    pieces = [
        *LABELS, "Bearer ", "bearer\t", "KBearer", "abc", "abcdefghij", " ", "  ", "\t", "&", '"',
        "'", "\\", "b'", "…", "[redacted]", REDACTED_VALUE, "A_TOKEN=", '\\"', '\\\\\\"', "x,",
    ]  # fmt: skip
    for _ in range(30_000):
        text = scrub("".join(rng.choice(pieces) for _ in range(rng.randint(1, 10))))
        cap = rng.randint(1, len(text) + 2)
        cut = bounded(text, cap, marker=marker)
        again = scrub(cut)
        assert len(again) <= cap, (text, cap, cut, again)
        if cut != text and cut:
            assert again.endswith(marker), (text, cap, cut, again)


def test_a_bound_over_many_labels_is_linear():
    started = time.monotonic()
    for text in (
        "?token= " * 40_000,
        "x-api-key: " * 30_000,
        "x-api-key:" + " " * 200_000 + "z",
        "?token=Bearer " * 30_000,
        "u/heartbeat/ " * 30_000,
        "a/heartbeat/" * 30_000 + " x",
        "&key=" * 60_000,
    ):
        bounded(text, len(text) - 1)
    assert time.monotonic() - started < 2.0


# --- a credential-named JSON member (issue #943) ------------------------------------------------

#: Every name on @origin's exact list, in each separator its words may take, and in each case.
MEMBER_NAMES = [
    *(
        spelled
        for first, second in (
            ("api", "key"),
            ("access", "token"),
            ("refresh", "token"),
            ("client", "secret"),
        )
        for joiner in ("_", "-", "")
        for spelled in (f"{first}{joiner}{second}", f"{first}{joiner}{second}".upper())
    ),
    *(
        spelled
        for name in ("password", "secret", "authorization")
        for spelled in (name, name.title())
    ),
]


def _nested(obj: object, depth: int) -> str:
    line = json.dumps(obj)
    for _ in range(depth):
        line = json.dumps({"log": line, "n": 1})
    return line


def _innermost(line: str, depth: int) -> dict:
    """`line` parsed back through every enclosing string to the object `_nested` began with."""
    obj = json.loads(line)
    for _ in range(depth):
        obj = json.loads(obj["log"])
    return obj


@pytest.mark.parametrize("name", MEMBER_NAMES)
@pytest.mark.parametrize("depth", [0, 1, 2])
def test_a_credential_named_member_loses_its_string_value_at_any_depth(name, depth):
    line = _nested({"id": 7, name: SHAPELESS_KEY, "k": "v"}, depth)
    scrubbed = scrub(line)
    assert SHAPELESS_KEY not in scrubbed
    assert _innermost(scrubbed, depth) == {"id": 7, name: "[redacted]", "k": "v"}


@pytest.mark.parametrize("separators", [(",", ":"), (", ", ": "), (",", " : "), (",\n", ":\t")])
def test_a_member_is_read_across_the_blanks_json_allows(separators):
    line = json.dumps({"password": SHAPELESS_KEY, "k": "v"}, separators=separators)
    assert json.loads(scrub(line)) == {"password": "[redacted]", "k": "v"}


@pytest.mark.parametrize("depth", [0, 1, 2])
@pytest.mark.parametrize(
    "value",
    [
        f'{SHAPELESS_KEY}"tail',  # an escaped quote inside the value
        f"{SHAPELESS_KEY}\\",  # an escaped backslash ending it, before the close
        f"{SHAPELESS_KEY}\\\\\\",
        f'\\"{SHAPELESS_KEY}\\\\"',
        f"{SHAPELESS_KEY}\n tail",  # an escaped newline, and a separator JSON writes raw
    ],
)
def test_a_members_value_is_read_to_its_own_close(depth, value):
    for ensure_ascii in (True, False):
        line = json.dumps({"secret": value, "k": "v"}, ensure_ascii=ensure_ascii)
        for _ in range(depth):
            line = json.dumps({"log": line}, ensure_ascii=ensure_ascii)
        scrubbed = scrub(line)
        assert SHAPELESS_KEY not in scrubbed, scrubbed
        assert _innermost(scrubbed, depth) == {"secret": "[redacted]", "k": "v"}, scrubbed


@pytest.mark.parametrize("value", [42, 3.5, None, True, {"a": "b"}, ["c"], ""])
def test_a_member_whose_value_is_no_string_is_left_alone(value):
    line = json.dumps({"api_key": value, "password": value, "k": "v"})
    assert scrub(line) == line


@pytest.mark.parametrize(
    "name",
    ["key", "token", "max_tokens", "total_tokens", "Key", "TOKEN", "my_api_key", "passwords",
     "secret_name", "my_password", "authorization_url", "api key", "api__key"],
)  # fmt: skip
def test_a_member_named_off_the_list_is_untouched(name):
    line = json.dumps({name: "kept-value-07", "n": 1})
    assert scrub(line) == line


def test_a_quote_after_an_even_run_never_opens_a_members_name():
    # The quote after `\\` is a real one (the backslashes escape each other), so it closes the
    # string before it rather than opening a member's name, and nothing is read.
    line = '{"h":"x\\\\"password":"kept"}'
    assert scrub(line) == line


def test_a_member_whose_value_never_closes_is_read_to_the_line_break():
    # A line cut inside the value, or a value in no JSON at all: the value runs on to the end of the
    # line, never past a raw line break, which no JSON string holds.
    assert scrub(f'{{"password":"{SHAPELESS_KEY}') == '{"password":"[redacted]'
    assert scrub(f'"password": "{SHAPELESS_KEY}\nnext') == '"password": "[redacted]\nnext'


def test_a_closing_quote_never_opens_the_next_name():
    # No JSON, but a second scrub must not read a name the first one did not: the backslashes the
    # value loses before its close once left that close free to open `"password"`.
    text = '"Secret" : "\\\\"password":"}'
    assert scrub(text) == '"Secret" : "[redacted]"password":"}'
    assert scrub(scrub(text)) == scrub(text)


def test_a_member_in_another_members_value_is_that_values_text():
    line = json.dumps({"password": '"secret": "x', "k": "v"})
    assert json.loads(scrub(line)) == {"password": "[redacted]", "k": "v"}


def test_a_python_repr_is_no_json_member():
    # Stated (issue #943): only `"` quotes a member, by the decision's statement.
    line = f"{{'password': '{SHAPELESS_KEY}'}}"
    assert scrub(line) == line


@pytest.mark.parametrize(
    ("text", "cap", "cut"),
    [
        # Cut inside the member's `[redacted]` or just after it, before its close: a second scrub
        # read the `…` into the value and wrote `[redacted]` past the cap, or dropped the `…`.
        ('{"n":1,"password":"[redacted]","k":"v"}', 20, '{"n":1,…'),
        ('{"n":1,"password":"[redacted]","k":"v"}', 30, '{"n":1,…'),
        ('{"n":1,"password":"[redacted]","k":"v"}', 31, '{"n":1,"password":"[redacted]"…'),
    ],
)
def test_a_bound_cuts_a_member_whole_when_the_marker_would_be_its_value(text, cap, cut):
    assert bounded(text, cap) == cut
    assert scrub(cut) == cut


def test_no_member_lets_a_second_scrub_grow_a_bound_or_take_its_marker():
    # Only texts a second scrub leaves as they are: where a value's own text holds a member whose
    # outer member another rule's span erases, a second scrub takes more with no cut at all (stated
    # in `docs/telemetry.md`), which is not the cut's doing.
    rng = random.Random(943)
    pieces = [
        '"password":"', '\\"api_key\\": \\"', '"Secret" : "', "abc", '"', '\\"', "\\\\", ",", "}",
        " ", "…", "[redacted]", "A_TOKEN=", "api_key=", "?token=",
    ]  # fmt: skip
    for _ in range(20_000):
        text = scrub("".join(rng.choice(pieces) for _ in range(rng.randint(1, 8))))
        if scrub(text) != text:
            continue
        cap = rng.randint(1, len(text) + 2)
        cut = bounded(text, cap)
        again = scrub(cut)
        assert len(again) <= cap, (text, cap, cut, again)
        if cut != text and cut:
            assert again.endswith("…"), (text, cap, cut, again)


def test_members_are_read_in_linear_time():
    started = time.monotonic()
    for text in (
        '"password":"' * 30_000,
        '{"password":"' + '\\\\\\"secret\\\\\\":\\\\\\"' * 20_000,
        '"secret":"' + "\\" * 200_000 + "x",
        '"secret":"' + '\\\\\\"' * 50_000,
        '{"password":"x",' * 30_000,
    ):
        scrub(text)
        bounded(text, len(text) - 1)
    assert time.monotonic() - started < 2.0
