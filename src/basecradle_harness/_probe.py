"""Recognize the NOC's `BCNOC2` marker: a real wake that runs as any other and is not mined.

The NOC ([basecradle-noc](https://github.com/basecradle/basecradle-noc)) posts this marker in the
body of a real platform message when it drives a verification wake of its own (a roll, a move, a
check on demand, basecradle-noc#1636). The wake answers it like any other message, model included;
the only difference a verified one makes is that the exchange is not handed to the memory provider
(`_wake.WakeAgent._observe`, issue #950).

It is the **verifying mirror** of basecradle-noc's ``src/basecradle_noc/marker.py`` (the sender
half). The two live in separate repos, and repo sovereignty forbids reaching across, so the scheme
is re-implemented here and the two halves **must agree byte for byte** on the format.

**The NOC's other marker, ``BCNOC1``, is not the harness's to verify** (issue #955). It is the
token-free synthetic probe, and it travels only on the router's synthetic route (the router's own
probe wake and its ack); no NOC code sends it to the platform. The harness used to verify a
``BCNOC1`` found in a message, a task, a webhook event or an asset and ack it before the model, and
because that marker is signed over its nonce alone, a copy of one valid marker in anyone's item
made a real item acked mechanically and never shown to the model. That path had no legitimate user
left, so it was removed: a ``BCNOC1`` in a platform item is ordinary content, claimed, answered by
the model, and mined.
"""

from __future__ import annotations

import hmac
import re
from hashlib import sha256

# Anchored, single-line; nonce and hmac are bounded token charsets so a marker can be located
# inside a larger body without catastrophic backtracking. Kept identical to basecradle-noc's
# marker.py: the two halves must agree byte for byte.
_NONCE = r"[A-Za-z0-9_-]{1,128}"


#: The marker of **a real wake that is not mined** (issue #950, basecradle-noc#1636). The NOC posts it
#: in a real platform message when it drives a verification wake of its own (a roll, a move, a check
#: on demand). It short-circuits nothing: the wake runs as any other, model included, and posts no
#: mechanical ack. The only difference a verified one makes is that the exchange is not handed to
#: the memory provider (`_wake.WakeAgent._observe`).
MARKED_SCHEME = "BCNOC2"
_MARKED_RE = re.compile(rf"\b{MARKED_SCHEME} (?P<nonce>{_NONCE}) (?P<hmac>[0-9a-f]{{64}})\b")


def marked_payload(nonce: str, timeline: str, author: str) -> bytes:
    """The bytes a `MARKED_SCHEME` marker signs: the scheme, the timeline, the author, the nonce.

    ``BCNOC2 <timeline-uuid> <author-uuid> <nonce>``, both uuids in canonical lowercase. Bound to
    **the timeline it was posted on and the account that posted it**, so a marker is good for one
    message's author on one timeline and nothing else: a copy pasted into a message on any other
    timeline, or into a message by anyone else on the same one, fails, and that message is mined
    like any other (issue #950). The signature proves the NOC minted the marker for this message's
    author here, not merely that it minted it once. The one place the signed bytes are decided, so
    the NOC's half (`marker.py`) and this one agree on one line.
    """
    return f"{MARKED_SCHEME} {_canonical(timeline)} {_canonical(author)} {nonce}".encode()


def _canonical(uuid: str) -> str:
    return uuid.strip().lower()


def verify_marked(body: str, secret: str, timeline: str, author: str) -> str | None:
    """The nonce of a valid `MARKED_SCHEME` marker for this ``timeline`` and ``author``, else ``None``.

    ``author`` is the uuid of the account that posted ``body``, as the platform asserts it.

    **Every** marker line in ``body`` is judged, each on its own, and the first that verifies is the
    answer: one check message may carry a line for each of several agents, and every agent but the
    first would otherwise read another agent's line and never its own. Accepting any is no wider
    than accepting the first, because each line is verified alone. The signature is compared in
    constant time; an unsigned, wrongly-signed, other-timeline or other-author marker is ``None``: an
    ordinary message in every respect. A ``BCNOC1`` marker never matches here.
    """
    for match in _MARKED_RE.finditer(body):
        payload = marked_payload(match.group("nonce"), timeline, author)
        expected = hmac.new(secret.encode(), payload, sha256).hexdigest()
        if hmac.compare_digest(expected, match.group("hmac")):
            return match.group("nonce")
    return None
