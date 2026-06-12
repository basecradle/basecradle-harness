"""Recognize a NOC synthetic-probe marker — the harness half of the message-seam contract.

The NOC ([basecradle-noc](https://github.com/basecradle/basecradle-noc)) drives the seam
*message posted → router wakes the recipient → agent replies* on a cadence and alerts when
the loop doesn't close. For that heartbeat to run **token-free at rest**, a woken harness
must recognize a NOC probe and ack it **at the reconcile layer, before any model call**
(see `_wake.py` → `_act_on`). This module is the recognition + ack half of that.

It is the **verifying mirror** of basecradle-noc's ``src/basecradle_noc/marker.py`` (the
sender half). The two live in separate repos — harness cannot import the NOC package, and
repo sovereignty forbids reaching across — so the scheme is re-implemented here, and the
two halves **MUST agree byte-for-byte** on the format. `marker.py` is the reference; keep
this identical to it (only the *verify* and *ack* sides are needed here — the harness never
mints a probe nor parses an ack).

The marker is a one-line, ASCII, HMAC-signed token embedded verbatim in the probe body::

    BCNOC1 <nonce> <hex-hmac-sha256>

- ``BCNOC1`` — scheme tag + version (bumpable without ambiguity).
- ``<nonce>`` — per-cycle correlation id the harness echoes in its ack so request and reply
  match exactly (opaque; conventionally uuid4 hex; never reused).
- ``<hmac>`` — hex ``HMAC-SHA256("BCNOC1 <nonce>", probe_secret)``, **constant-time compared.**

**Why signed, not a bare sentinel.** The short-circuit fires *before the model*, so a
forgeable marker would let any peer (a) spend the fleet's free-ack path at will and (b) — far
worse — get a *real* message mistaken for a probe and silently never answered, the exact
silent-death class the NOC exists to catch, manufactured on demand. Only a holder of the
shared ``probe_secret`` can mint a valid marker. The secret lives in neither repo; it is
provisioned to the NOC box and the harness out of band and read from the environment.
"""

from __future__ import annotations

import hmac
import re
from hashlib import sha256

SCHEME = "BCNOC1"
ACK_TAG = f"{SCHEME}-ACK"

# Anchored, single-line; nonce and hmac are bounded token charsets so a marker can be
# located inside a larger body without catastrophic backtracking. Kept identical to
# basecradle-noc's marker.py — the two halves must agree byte-for-byte.
_NONCE = r"[A-Za-z0-9_-]{1,128}"
_MARKER_RE = re.compile(rf"\b{SCHEME} (?P<nonce>{_NONCE}) (?P<hmac>[0-9a-f]{{64}})\b")


def _signature(nonce: str, secret: str) -> str:
    """The hex HMAC-SHA256 over "SCHEME nonce", keyed by the shared probe secret."""
    payload = f"{SCHEME} {nonce}".encode()
    return hmac.new(secret.encode(), payload, sha256).hexdigest()


def verify_probe(body: str, secret: str) -> str | None:
    """Return the nonce if ``body`` carries a valid, correctly-signed marker, else ``None``.

    The recognition the harness performs at the reconcile layer to decide whether to
    short-circuit (a token-free ack) instead of calling the model. The signature is
    compared in constant time; an unsigned or wrongly-signed sentinel is rejected, which
    is the whole point of the scheme. A body with no marker at all returns ``None`` — an
    ordinary message that must fall through to the model.
    """
    match = _MARKER_RE.search(body)
    if match is None:
        return None
    expected = _signature(match.group("nonce"), secret)
    if hmac.compare_digest(expected, match.group("hmac")):
        return match.group("nonce")
    return None


def ack_line(nonce: str) -> str:
    """The deterministic ack line the harness posts to close the loop.

    The ack is **not** signed: its authenticity is its author. The NOC prober accepts an
    ack only when it both carries the matching nonce **and** was posted by the woken party
    (@jt) — and that pairing is exactly what the seam verifies ("the message woke @jt and
    @jt replied"). A signature would prove only that some secret-holder wrote it, which
    says nothing about whether the wake actually arrived.
    """
    return f"{ACK_TAG} {nonce}"
