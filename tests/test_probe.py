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
