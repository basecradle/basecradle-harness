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
AUTHOR = "019e7750-66ee-7e50-9e54-3bf8c3d6a8f1"  # the account that posts the check message
OTHER_AUTHOR = "019e7756-9f60-7a80-93a4-6f7081920314"
# The pinned vector for BCNOC2: HMAC-SHA256 over "BCNOC2 <timeline> <author> <nonce>" under SECRET.
# The NOC's mint must produce exactly this; do not regenerate without the NOC side agreeing on #950.
MARKED_HMAC = "3440fafd154ebf299e6873ae65d8bb85d6917c469ce41a99ab05e7f759cb2008"
MARKED = f"BCNOC2 {NONCE} {MARKED_HMAC}"


def marked(nonce, timeline=TIMELINE, author=AUTHOR, secret=SECRET):
    sig = hmac.new(secret.encode(), marked_payload(nonce, timeline, author), sha256).hexdigest()
    return f"BCNOC2 {nonce} {sig}"


def _signed_over(text):
    return f"BCNOC2 {NONCE} {hmac.new(SECRET.encode(), text.encode(), sha256).hexdigest()}"


def test_the_marked_pinned_vector_verifies_for_its_timeline_and_author():
    assert marked_payload(NONCE, TIMELINE, AUTHOR) == f"BCNOC2 {TIMELINE} {AUTHOR} {NONCE}".encode()
    assert verify_marked(f"NOC check.\n{MARKED}", SECRET, TIMELINE, AUTHOR) == NONCE


def test_a_copy_on_another_timeline_or_by_another_author_does_not_verify():
    """A real marker pasted anywhere else is an ordinary message: the bytes name both."""
    assert verify_marked(MARKED, SECRET, TIMELINE, AUTHOR) == NONCE
    assert verify_marked(MARKED, SECRET, OTHER_TIMELINE, AUTHOR) is None
    assert verify_marked(MARKED, SECRET, TIMELINE, OTHER_AUTHOR) is None


def test_a_marker_signed_without_its_timeline_or_its_author_never_verifies():
    """A mutation that drops either binding is caught here, not only by a copy failing."""
    for text in (
        f"BCNOC2 {NONCE}",
        f"BCNOC2 {TIMELINE} {NONCE}",
        f"BCNOC2 {AUTHOR} {NONCE}",
    ):
        for timeline, author in ((TIMELINE, AUTHOR), (OTHER_TIMELINE, OTHER_AUTHOR)):
            assert verify_marked(_signed_over(text), SECRET, timeline, author) is None


def test_the_uuids_are_read_in_their_canonical_form():
    assert verify_marked(MARKED, SECRET, f" {TIMELINE.upper()} ", AUTHOR.upper()) == NONCE


def test_a_wrongly_signed_or_misshapen_marked_line_does_not_verify():
    assert verify_marked(marked(NONCE, secret="another-secret"), SECRET, TIMELINE, AUTHOR) is None
    assert verify_marked(f"BCNOC2 {NONCE} {'0' * 64}", SECRET, TIMELINE, AUTHOR) is None
    assert verify_marked(f"BCNOC2 {NONCE}", SECRET, TIMELINE, AUTHOR) is None
    assert verify_marked("", SECRET, TIMELINE, AUTHOR) is None


def test_every_marker_line_is_judged_on_its_own():
    """A forged or another agent's line first never hides a valid one after it."""
    other = marked("ffff", secret="another-agents-secret")
    assert verify_marked(f"{other}\n{MARKED}", SECRET, TIMELINE, AUTHOR) == NONCE
    assert verify_marked(f"BCNOC2 {NONCE} {'0' * 64}\n{MARKED}", SECRET, TIMELINE, AUTHOR) == NONCE
    assert verify_marked(f"{other}\n{other}", SECRET, TIMELINE, AUTHOR) is None


def test_the_two_kinds_never_stand_in_for_each_other():
    """A BCNOC2 is never acked as a probe, and a BCNOC1 never marks a real wake."""
    assert verify_probe(MARKED, SECRET) is None
    assert verify_marked(MARKER, SECRET, TIMELINE, AUTHOR) is None
    assert verify_probe(MARKED.replace("BCNOC2", "BCNOC1"), SECRET) is None
