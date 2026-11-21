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

- **Does another quick try follow?** (`another_try_follows`, issue #962) Inside one attempt the router
  runs an exit ``1`` again up to its quick-try bound (basecradle-router#533), and says which try this
  is through ``BASECRADLE_WAKE_TRY`` / ``BASECRADLE_WAKE_TRIES``. An exit ``1`` another try follows is
  **deferred** too, by the same rule: no ``wake failed``, and ``wake end … outcome=deferred`` carries
  the error. An exit ``75`` is never quick-retried inside a real attempt, so it asks the attempt
  pair; an exit ``1`` asks the try pair, and that is the whole difference between the two.

The two are not complements. A wake with neither variable (run by hand, by a router that predates
them) and a wake whose pair is broken answer **no** to both: such a wake posts no notice, as it did
before #951, and an outage it gives up on is the failure it always was, because nothing promises a
next attempt. So the error each fallback costs is a notice not posted, or a page that was not
needed; never a notice posted early, and never a failure left unpaged. The try pair falls back the same way: no pair, or a broken one, is no try after
this one.
"""

from __future__ import annotations

import logging
import os

_log = logging.getLogger("basecradle_harness")

WAKE_ATTEMPT_ENV = "BASECRADLE_WAKE_ATTEMPT"
WAKE_ATTEMPTS_ENV = "BASECRADLE_WAKE_ATTEMPTS"
WAKE_TRY_ENV = "BASECRADLE_WAKE_TRY"
WAKE_TRIES_ENV = "BASECRADLE_WAKE_TRIES"

#: The broken pairs already reported, so a wake that asks several times says it once.
_reported: set[tuple[str, str, str]] = set()


def _pair(number_env: str, count_env: str, *, what: str, reading: str) -> tuple[int, int] | None:
    """``(number, count)`` as the router exported them, or ``None`` when they say nothing.

    Neither variable set is ``None`` silently. A value that is not a positive integer, or only one
    of the pair, is the router's contract broken, so it is logged (once per process for the same
    values: a wake asks from the engine, the wake and the entrypoint) and read as ``None`` too.
    """
    raw_number = (os.environ.get(number_env) or "").strip()
    raw_count = (os.environ.get(count_env) or "").strip()
    if not raw_number and not raw_count:
        return None
    try:
        number, count = int(raw_number), int(raw_count)
    except ValueError:
        number = count = 0
    if number < 1 or count < 1:
        if (number_env, raw_number, raw_count) in _reported:
            return None
        _reported.add((number_env, raw_number, raw_count))
        _log.warning(
            "The router's %s variables are not a pair of positive integers "
            "(%s=%r, %s=%r), so this wake is read as %s.",
            what,
            number_env,
            raw_number,
            count_env,
            raw_count,
            reading,
        )
        return None
    return number, count


def _attempt_pair() -> tuple[int, int] | None:
    """``(attempt, attempts)``: which of the delivery's attempts this wake is (issue #951)."""
    return _pair(
        WAKE_ATTEMPT_ENV,
        WAKE_ATTEMPTS_ENV,
        what="attempt",
        reading="neither the last attempt nor one another attempt follows",
    )


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


def another_try_follows() -> bool:
    """Will the router run this wake again, inside the same attempt, if it exits ``1``? (issue #962)

    ``BASECRADLE_WAKE_TRY < BASECRADLE_WAKE_TRIES`` (basecradle-router#533). No pair, or a broken
    one, answers no, so the ending stays the failure it was: a retry is never guessed. A probe
    carries ``1``/``1``, so it never reads as having a try after it.
    """
    pair = _pair(WAKE_TRY_ENV, WAKE_TRIES_ENV, what="try", reading="the last try")
    return pair is not None and pair[0] < pair[1]


def retried(*, outage: bool) -> bool:
    """Will the router run this wake again after the exit it is about to make? (issues #957, #962)

    An outage exits ``75``, which the router answers with its next attempt and never with a quick
    try; every other fault exits ``1``, which the router answers with its next quick try and never
    with an attempt. One function, so the entrypoint and the wake judge the same exit the same way.
    """
    return another_attempt_follows() if outage else another_try_follows()
