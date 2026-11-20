"""The NOC synthetic-probe marker — the harness's verifying half (issue #106).

`_probe` is the recognition mirror of basecradle-noc's `marker.py`; the two halves live
in separate repos and **must agree byte-for-byte**, so the centerpiece here is a *pinned
vector* — a literal (secret, nonce, hmac) triple computed by the shared scheme. If the
scheme ever drifts (a changed payload, key handling, or tag), this test breaks loudly.

The wake-level behavior (ack token-free, no model call, clean transcript) is pinned in
`test_wake.py`; this file pins the marker math.
"""

import hmac
from hashlib import sha256

from basecradle_harness._probe import ack_line, verify_probe

# A pinned vector — the literal HMAC the shared scheme produces for this (secret, nonce).
# Locks byte-compatibility with basecradle-noc's marker.py; do not regenerate to "fix" a
# failure without confirming the NOC side changed too.
SECRET = "noc-probe-secret-do-not-use-in-prod"
NONCE = "0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d"
HMAC = "b7848d0c0573e3230ccff00ceca709604dba517b00a440e68915267e542e51a9"
MARKER = f"BCNOC1 {NONCE} {HMAC}"


def signed(nonce, secret=SECRET):
    """A correctly-signed marker line for `nonce` under `secret` (mirrors marker.mint)."""
    sig = hmac.new(secret.encode(), f"BCNOC1 {nonce}".encode(), sha256).hexdigest()
    return f"BCNOC1 {nonce} {sig}"


def test_pinned_vector_verifies():
    """The literal scheme output for a known (secret, nonce) verifies to that nonce."""
    assert verify_probe(MARKER, SECRET) == NONCE


def test_a_freshly_signed_marker_verifies():
    assert verify_probe(signed("feedfacecafe"), SECRET) == "feedfacecafe"


def test_marker_embedded_in_a_human_legible_body_is_found():
    """The probe body is a legible line plus the marker; the marker is located inside it."""
    body = f"NOC message-seam probe — please disregard.\n{MARKER}\n"
    assert verify_probe(body, SECRET) == NONCE


def test_a_forged_hmac_is_rejected():
    """Right shape, wrong signature → not a probe (the whole point of signing)."""
    forged = f"BCNOC1 {NONCE} {'0' * 64}"
    assert verify_probe(forged, SECRET) is None


def test_the_wrong_secret_is_rejected():
    assert verify_probe(MARKER, "not-the-probe-secret") is None


def test_a_body_with_no_marker_is_not_a_probe():
    assert verify_probe("just an ordinary message, nothing to see here", SECRET) is None


def test_a_bare_unsigned_sentinel_is_rejected():
    """A `BCNOC1 <nonce>` with no hmac field does not match — forgery by omission fails."""
    assert verify_probe(f"BCNOC1 {NONCE}", SECRET) is None


def test_ack_line_echoes_the_nonce_unsigned():
    assert ack_line(NONCE) == f"BCNOC1-ACK {NONCE}"


# --- BCNOC2: the marked real wake (issue #950) ---------------------------------------------------

from basecradle_harness._probe import marked_payload, verify_marked  # noqa: E402

TIMELINE = "019e7750-66ee-7f53-829f-13a8a710b6da"
OTHER_TIMELINE = "019e7750-66ee-7f53-829f-13a8a710b6db"
# The pinned vector for BCNOC2: HMAC-SHA256 over "BCNOC2 <timeline> <nonce>" under SECRET. The NOC's
# mint must produce exactly this; do not regenerate without the NOC side agreeing on #950.
MARKED_HMAC = "26a9203bd55f36a47731f03315514e40930eced3ec72f14377922a41c1a3400c"
MARKED = f"BCNOC2 {NONCE} {MARKED_HMAC}"


def marked(nonce, timeline=TIMELINE, secret=SECRET):
    sig = hmac.new(secret.encode(), marked_payload(nonce, timeline), sha256).hexdigest()
    return f"BCNOC2 {nonce} {sig}"


def test_the_marked_pinned_vector_verifies_on_its_timeline():
    assert marked_payload(NONCE, TIMELINE) == f"BCNOC2 {TIMELINE} {NONCE}".encode()
    assert verify_marked(f"NOC check.\n{MARKED}", SECRET, TIMELINE) == NONCE


def test_a_marked_copy_on_another_timeline_does_not_verify():
    """A real marker pasted elsewhere is an ordinary message: nobody can make an agent forget one."""
    assert verify_marked(MARKED, SECRET, TIMELINE) == NONCE
    assert verify_marked(MARKED, SECRET, OTHER_TIMELINE) is None


def test_a_marker_signed_without_its_timeline_never_verifies():
    """The signed bytes name the timeline; a line signed over "BCNOC2 <nonce>" alone is forged."""
    unbound = hmac.new(SECRET.encode(), f"BCNOC2 {NONCE}".encode(), sha256).hexdigest()
    for timeline in (TIMELINE, OTHER_TIMELINE):
        assert verify_marked(f"BCNOC2 {NONCE} {unbound}", SECRET, timeline) is None


def test_the_timeline_is_read_as_the_harness_records_it():
    assert verify_marked(MARKED, SECRET, f" {TIMELINE.upper()} ") == NONCE


def test_a_wrongly_signed_or_misshapen_marked_line_does_not_verify():
    assert verify_marked(marked(NONCE, secret="another-secret"), SECRET, TIMELINE) is None
    assert verify_marked(f"BCNOC2 {NONCE} {'0' * 64}", SECRET, TIMELINE) is None
    assert verify_marked(f"BCNOC2 {NONCE}", SECRET, TIMELINE) is None
    assert verify_marked("", SECRET, TIMELINE) is None


def test_the_two_kinds_never_stand_in_for_each_other():
    """A BCNOC2 is never acked as a probe, and a BCNOC1 never marks a real wake."""
    assert verify_probe(MARKED, SECRET) is None
    assert verify_marked(MARKER, SECRET, TIMELINE) is None
    forged_as_probe = MARKED.replace("BCNOC2", "BCNOC1")
    assert verify_probe(forged_as_probe, SECRET) is None
