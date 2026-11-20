"""Recognize a NOC synthetic-probe marker — the harness half of the NOC seam contracts.

The NOC ([basecradle-noc](https://github.com/basecradle/basecradle-noc)) drives each
wake seam — *message posted*, *webhook delivered*, and *task activated*, each → *router
wakes the recipient → agent acts* — on a cadence and alerts when a loop doesn't close. For
those heartbeats to run **token-free at rest**, a woken harness must recognize a NOC probe
and ack it **at the reconcile layer, before any model call** (see `_wake.py` → `_act_on`,
whose `probe=` hook all three reconcilers pass). The marker is identical across seams; only
the carrier field differs (a message body, a webhook payload, a task's instructions). This
module is the recognition + ack half of all three contracts, and the recognition half of a second
kind of marker, `MARKED_SCHEME` (``BCNOC2``): a real wake that runs as any other and is not mined
(issue #950). That one is bound to the timeline it was posted on (`marked_payload`).

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


#: The second kind of the signed marker (issue #950, basecradle-noc#1636): **a real wake that is not
#: mined.** The NOC posts it in a real platform message when it drives a verification wake of its
#: own (a roll, a move, a check on demand). Unlike `SCHEME` it short-circuits nothing: the wake runs
#: as any other, model included, and posts no mechanical ack. The only difference a verified one
#: makes is that the exchange is not handed to the memory provider (`_wake.WakeAgent._observe`).
MARKED_SCHEME = "BCNOC2"
_MARKED_RE = re.compile(rf"\b{MARKED_SCHEME} (?P<nonce>{_NONCE}) (?P<hmac>[0-9a-f]{{64}})\b")


def marked_payload(nonce: str, timeline: str) -> bytes:
    """The bytes a `MARKED_SCHEME` marker signs: the scheme, **the timeline it was posted on**, the nonce.

    Bound to the timeline so a marker copied into a message on any other timeline fails, and that
    message is mined like any other: the signature proves the NOC minted the marker for the
    timeline the harness is reading, not merely that it minted it once (issue #950). The one place
    the signed bytes are decided, so the NOC's half (`marker.py`) and this one agree on one line.
    """
    return f"{MARKED_SCHEME} {timeline.strip().lower()} {nonce}".encode()


def verify_marked(body: str, secret: str, timeline: str) -> str | None:
    """Return the nonce if ``body`` carries a valid `MARKED_SCHEME` marker for ``timeline``, else ``None``.

    The first marker in ``body`` is the one judged, as `verify_probe` judges its first. The
    signature is compared in constant time; an unsigned, wrongly-signed or other-timeline marker is
    ``None``: an ordinary message in every respect. A `SCHEME` marker never matches here.
    """
    match = _MARKED_RE.search(body)
    if match is None:
        return None
    expected = hmac.new(
        secret.encode(), marked_payload(match.group("nonce"), timeline), sha256
    ).hexdigest()
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
