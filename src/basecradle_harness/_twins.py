"""The twin rule: an exchange already filed, word for word, is not filed again (issue #950).

**Routine exchanges are not memories** (`@origin`, 2026-11-20, basecradle#633). Measured on one
agent's store, 427 drawers (7.5% of all) were routine check-ins near-identical to each other: 42
of them unreachable by search because about 385 twins outranked them, and every one of them
competing with a distinctive memory for a place in recall. The agent keeps its transcript; memory
is for what was learned.

**The rule keys on what was said, never on how the wake started.** A scheduled check-in can be the
wake on which the agent learned something, so a rule that recognized routine *kinds* of turn (a
task activation, an hourly heartbeat) would throw that wake away with the rest. This one needs no
list of kinds: an exchange is a twin when its words are the words of an exchange already filed,
once the harness's own stamps are set aside, so the next routine shape nobody has thought of is
covered too.

**Exact, never near.** A near-twin rule (a similarity threshold over tokens, shingles or an
embedding) cannot be made safe mechanically: "the deploy is ready" and "the deploy is not ready"
are near-twins by every such measure, and the second can be the one fact a scheduled wake learned.
Only an exchange whose words are exactly those of one already filed provably adds nothing a memory
does not hold. A routine reply the model phrases a new way is filed, once per phrasing; a routine
reply it repeats is not.

**What "the harness's own stamps" means** is exactly what the dialogue renderers (`_wake`) write
from an item's metadata, which differs on every repetition of the same words:

- the ``[created_at]`` stamp that opens each item's line (a message, a batch's every message, an
  asset, a webhook delivery, a task activation);
- a task activation's own uuid and send time (``(task <uuid>, scheduled for <time>)``);
- a webhook delivery's own event uuid (``(event <uuid>, endpoint``).

Each is masked only **where the renderer puts it**, never wherever such a shape appears: a uuid or
a time a peer writes in a message is what was said, and a message that differs from an earlier one
only in that value is a different message. The handle, an endpoint's uuid and its author are kept:
they are who spoke, which a memory must keep. Whitespace runs count as one space. Nothing else is
normalized: case and punctuation are words too ("$5" and "€5").

**The key is the engine's; the record is the store's.** `twin_key` is computed at the one place the
harness mines (`_wake.WakeAgent._observe`) for every provider, rides the exchange
(`MemoryExchange.twin`), and the provider answers from what it actually holds
(`MemoryProvider.holds`). There is no side list, so an exchange removed from the store (a scrub, a
wiped palace) is filed again the next time it happens, and nothing already filed is touched.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass

#: The rule's version, carried in every key. A key made under another version never equals one
#: made under this one, so changing the rule can only ever file an exchange once more, never
#: suppress one the new rule would keep.
TWIN_RULE = 1

#: A platform moment as the renderers print it: ISO 8601 with a ``T`` (or, for a datetime the SDK
#: hands over as an object, a space), optional fractional seconds, and a ``Z`` or an offset.
_MOMENT = r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?"
_UUID = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"

#: Each stamp a dialogue renderer writes, matched only where it writes it, and what stands in for
#: it. `tests/test_twins.py` drives every renderer twice with only the stamps changed and asserts
#: one key, so a reworded template fails the build rather than silently filing every repetition.
_STAMPS = (
    (re.compile(rf"^\[{_MOMENT}\] ", re.MULTILINE), "[*] "),
    (re.compile(rf"\(task {_UUID}, scheduled for {_MOMENT}\)"), "(task *, scheduled for *)"),
    (re.compile(rf"\(event {_UUID}, endpoint "), "(event *, endpoint "),
)


def twin_key(user: str, assistant: str) -> str:
    """The exchange's twin key: equal for two exchanges exactly when they are twins.

    The two halves stay apart (a reply that moved into the peer's half is a different exchange),
    and the key names `TWIN_RULE`. A SHA-256 hex digest, so the key says nothing about what was
    said: a store keeps it beside the text it already holds, and a log line could carry it.
    """
    digest = hashlib.sha256(f"{_canonical(user)}\x00{_canonical(assistant)}".encode()).hexdigest()
    return f"v{TWIN_RULE}:{digest}"


def _canonical(text: str) -> str:
    """``text`` with the renderers' stamps masked and every whitespace run one space."""
    for pattern, mask in _STAMPS:
        text = pattern.sub(mask, text)
    return " ".join(text.split())


@dataclass(frozen=True)
class TwinCensus:
    """What the rule does to a recorded history.

    ``kept`` and ``skipped`` partition the exchanges read; ``collapsed`` is how many of the kept
    ones the skipped were twins of, so ``skipped`` repetitions collapse onto ``collapsed`` memories.
    """

    kept: int
    skipped: int
    collapsed: int

    @property
    def total(self) -> int:
        return self.kept + self.skipped


def census(exchanges: Iterable[tuple[str, str]]) -> TwinCensus:
    """Replay ``exchanges`` (``(user, assistant)``, in the order they were filed) through the rule.

    The first of each set of twins is kept and every later one skipped, which is what the rule
    would have done had it been in force from the first exchange. Nothing is read or written but
    the exchanges themselves.
    """
    seen: set[str] = set()
    repeated: set[str] = set()
    kept = skipped = 0
    for user, assistant in exchanges:
        key = twin_key(user, assistant)
        if key in seen:
            skipped += 1
            repeated.add(key)
        else:
            seen.add(key)
            kept += 1
    return TwinCensus(kept=kept, skipped=skipped, collapsed=len(repeated))
