"""The repeat rule: an exchange that repeats the newest one filed on its timeline is not filed (#950).

**Routine exchanges are not memories** (`@origin`, 2026-11-20, basecradle#633). Measured on one
agent's store, 427 drawers (7.5% of all) were routine check-ins near-identical to each other: 42
of them unreachable by search because about 385 twins outranked them, and every one of them
competing with a distinctive memory for a place in recall. The agent keeps its transcript; memory
is for what was learned.

**The rule keys on what was said, never on how the wake started.** A scheduled check-in can be the
wake on which the agent learned something, so a rule that recognized routine *kinds* of turn (a
task activation, an hourly heartbeat) would throw that wake away with the rest. This one needs no
list of kinds, so the next routine shape nobody has thought of is covered too.

**What "adds nothing" means, exactly.** An exchange adds nothing when its words are the words of the
**newest exchange already filed on the same timeline**, the harness's opening stamp aside. Each
part of that sentence closes a way of losing what was learned:

- **Exact, never near.** A near-twin rule (a similarity threshold over tokens, shingles or an
  embedding) cannot be made safe mechanically: "the deploy is ready" and "the deploy is not ready"
  are near-twins by every such measure, and the second can be the one fact a scheduled wake
  learned. A routine reply the model phrases a new way is filed; one it repeats is not.
- **The newest, never any.** Memory is read as "later is current". A status that went ``ok``, then
  ``down``, then ``ok`` again would, under an any-twin rule, lose its third exchange, and memory's
  newest record would say ``down``. A repeat of the newest record changes neither what memory holds
  nor which record is newest.
- **On its own timeline.** Recall says where a memory was said. The same words on another timeline
  are another event, and are filed.

**The stamp is masked only where the renderer writes it: the opening of the peer's half.** Every
dialogue renderer (`_wake._dialogue_of`, `_wake._render_batch`) opens with the item's
``[created_at]`` stamp; a task activation's header follows it with the task's own uuid and send
time, and a webhook delivery's with its event uuid. Those differ on every repetition of the same
words, so they are set aside, at that one position and nowhere else: never in the reply, never in
the body of what a peer wrote, so a time or a uuid anyone *said* is what was said. (A batch's later
messages keep their stamps, so a repeated batch of several messages is filed: the safe direction.)
Leading and trailing whitespace is dropped; nothing else is normalized, indentation and case and
punctuation included ("$5" and "€5").

**The key is the engine's; the record is the store's.** `twin_key` is computed at the one place the
harness mines (`_wake.WakeAgent._observe`) for every provider, rides the exchange
(`MemoryExchange.twin`), and the provider answers from what it actually holds
(`MemoryProvider.repeats`). There is no side list, so an exchange removed from the store (a scrub,
a wiped palace) is filed again the next time it happens, and nothing already filed is touched.
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

#: The openings a dialogue renderer writes, each anchored at the start of the peer's half and tried
#: in order, the first match masked. The task and delivery headers come first because they open with
#: the same stamp. `tests/test_twins.py` drives every renderer twice with only its stamps changed and
#: asserts one key, so a reworded template fails the build rather than silently filing every repeat.
_OPENINGS = (
    (
        re.compile(
            rf"\A\[{_MOMENT}\] (A task scheduled by @\S+ has activated and is now in front of every "
            rf"viewer of this timeline )\(task {_UUID}, scheduled for {_MOMENT}\)"
        ),
        r"[*] \1(task *, scheduled for *)",
    ),
    (
        re.compile(
            rf"\A\[{_MOMENT}\] (An inbound webhook was delivered to this timeline )"
            rf"\(event {_UUID}, endpoint "
        ),
        r"[*] \1(event *, endpoint ",
    ),
    (re.compile(rf"\A\[{_MOMENT}\] "), "[*] "),
)


def twin_key(user: str, assistant: str) -> str:
    """The exchange's key: equal for two exchanges exactly when their words are the same.

    Only the opening of ``user`` is masked (`_OPENINGS`); ``assistant`` is taken as written. The two
    halves are length-prefixed, so no text in either can move the boundary between them, and the
    key names `TWIN_RULE`. A SHA-256 hex digest, so the key says nothing about what was said.
    """
    user = _opening_masked(user.strip())
    assistant = assistant.strip()
    payload = f"{len(user)}:{user}{len(assistant)}:{assistant}"
    return f"v{TWIN_RULE}:{hashlib.sha256(payload.encode()).hexdigest()}"


def _opening_masked(user: str) -> str:
    for pattern, mask in _OPENINGS:
        masked, count = pattern.subn(mask, user, count=1)
        if count:
            return masked
    return user


@dataclass(frozen=True)
class TwinCensus:
    """What the rule does to a recorded history.

    ``kept`` and ``skipped`` partition the exchanges read; ``collapsed`` is how many kept records
    the skipped ones repeat (one record per timeline and key).
    """

    kept: int
    skipped: int
    collapsed: int

    @property
    def total(self) -> int:
        return self.kept + self.skipped


def census(exchanges: Iterable[tuple[str | None, str, str]]) -> TwinCensus:
    """Replay ``(timeline, user, assistant)`` exchanges, in filing order, through the rule.

    An exchange is skipped when its key is the key of the newest exchange kept on its timeline, which
    is what the rule would have done had it been in force from the first exchange. An exchange with
    no timeline is always kept, as the observe keeps it. Nothing is read or written but the
    exchanges themselves.
    """
    newest: dict[str, str] = {}
    repeated: set[tuple[str, str]] = set()
    kept = skipped = 0
    for timeline, user, assistant in exchanges:
        key = twin_key(user, assistant)
        if timeline is not None and newest.get(timeline) == key:
            skipped += 1
            repeated.add((timeline, key))
            continue
        kept += 1
        if timeline is not None:
            newest[timeline] = key
    return TwinCensus(kept=kept, skipped=skipped, collapsed=len(repeated))
