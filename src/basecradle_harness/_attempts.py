"""Which of the router's attempts at one delivery this wake is (issues #951, #957).

A wake that gives up on an outage exits ``75`` (`_wake.EX_TEMPFAIL`), and the router answers that
exit alone with its patient schedule: four attempts in all, at about 1, 5 and 30 minutes
(basecradle-router#527). Each attempt is told which it is through two variables, set on every real
wake (never on a probe) and exported after `agent.env`.

Two questions are read off them, and each has one reader of consequence:

- **Is this the last attempt?** (`last_attempt`) Only the last one tells the timeline about an
  outage: every earlier attempt may simply succeed on the next try.
- **Does another attempt follow?** (`another_attempt_follows`) Then an outage this wake gives up on
  is **deferred**, not failed: the wake ends ``outcome=deferred``, and nothing it writes about the
  outage is an ERROR, because the router is about to try again (`@origin`, 2026-11-20: *"'fail'
  shouldn't alarm until it has failed! NOT while it is still trying!"*).

The two are not complements. A wake with neither variable (run by hand, by a router that predates
them) and a wake whose pair is broken answer **no** to both: such a wake posts no notice, as it did
before #951, and an outage it gives up on is the failure it always was, because nothing promises a
next attempt. So the error each fallback costs is a notice not posted, or a page that was not
needed; never a notice posted early, and never a failure left unpaged.
"""

from __future__ import annotations

import logging
import os

_log = logging.getLogger("basecradle_harness")

WAKE_ATTEMPT_ENV = "BASECRADLE_WAKE_ATTEMPT"
WAKE_ATTEMPTS_ENV = "BASECRADLE_WAKE_ATTEMPTS"

#: The broken pairs already reported, so a wake that asks several times says it once.
_reported: set[tuple[str, str]] = set()


def _attempt_pair() -> tuple[int, int] | None:
    """``(attempt, attempts)`` as the router exported them, or ``None`` when they say nothing.

    Neither variable set is ``None`` silently. A value that is not a positive integer, or only one
    of the pair, is the router's contract broken, so it is logged (once per process for the same
    values: a wake asks from the engine, the wake and the entrypoint) and read as ``None`` too.
    """
    raw_attempt = (os.environ.get(WAKE_ATTEMPT_ENV) or "").strip()
    raw_attempts = (os.environ.get(WAKE_ATTEMPTS_ENV) or "").strip()
    if not raw_attempt and not raw_attempts:
        return None
    try:
        attempt, attempts = int(raw_attempt), int(raw_attempts)
    except ValueError:
        attempt = attempts = 0
    if attempt < 1 or attempts < 1:
        if (raw_attempt, raw_attempts) in _reported:
            return None
        _reported.add((raw_attempt, raw_attempts))
        _log.warning(
            "The router's attempt variables are not a pair of positive integers "
            "(%s=%r, %s=%r), so this wake is read as neither the last attempt nor one "
            "another attempt follows.",
            WAKE_ATTEMPT_ENV,
            raw_attempt,
            WAKE_ATTEMPTS_ENV,
            raw_attempts,
        )
        return None
    return attempt, attempts


def last_attempt() -> bool:
    """Is this wake the router's **last** attempt at its delivery? (issue #951)

    ``BASECRADLE_WAKE_ATTEMPT == BASECRADLE_WAKE_ATTEMPTS``, and only then does a wake that gave up
    on an outage tell the timeline. An attempt past the count is read as last, since there is no
    attempt after it. No pair, or a broken one, is not a last attempt.
    """
    pair = _attempt_pair()
    return pair is not None and pair[0] >= pair[1]


def another_attempt_follows() -> bool:
    """Will the router try this delivery again if this wake exits ``75``? (issue #957)

    ``BASECRADLE_WAKE_ATTEMPT < BASECRADLE_WAKE_ATTEMPTS``. No pair, or a broken one, answers no:
    nothing promises the attempt, so the outage stays the failure it was.
    """
    pair = _attempt_pair()
    return pair is not None and pair[0] < pair[1]
