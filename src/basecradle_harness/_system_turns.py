"""Where a system turn may travel in the system role, and how it reads where it may not (issue #945).

The harness places system turns all through a conversation, not only at its head: the per-wake
brief (spliced immediately before the newest user turn), the per-step note (`Current Time: … /
Step N of M.`, appended before every model call and kept in the transcript as the step ledger), the
reserve nudge, a truncation or failure note, a compaction summary. Most models take every one of
them in place, in the system role, and for those the wire is left exactly as it is.

**Some models, as some vendors serve them, refuse a system-role message that is not followed by a
user turn.** The first member of that category is refuseless's ``kimi-k3.1-pro-abliterated``,
measured live by the capital on 2026-11-20 on both of the vendor's surfaces: ``developer, user``,
``developer, developer, user`` and ``user, developer, user`` answer 200; ``user, developer``,
``developer, user, developer`` and ``developer, user, assistant, developer`` answer 400 with a
generic ``invalid_request``, and so does the order every wake sends (``user, developer (the brief),
user, developer (the note)``). The same request with the trailing note sent as a ``user`` item is
200, and so is every other order on ``deepseek-v4.1-flash-abliterated-fast`` at the same vendor, so
this is how the vendor serves one model, not a property of the vendor.

The fact is **declared, never detected** (``AI_SYSTEM_TURNS=before_user``, read in
`_basecradle.system_turns_from_env`). The vendor's refusal names nothing (*"Invalid or unsupported
request parameters"*), so detecting it would mean re-sending every request a vendor refuses with a
400 in a second shape, on every agent; an agent whose model takes today's shape would then send a
request it never sent before, and the one it was refused for would be answered by a guess. Nor is it
a table of model names here, which would be a measurement with a date on it (`CLAUDE.md` → Provider
Capabilities, the vendor-cap-table rule). The operator who chose the model declares how it is
served, beside ``AI_MODEL``.

`before_user` is the whole rule, and it is narrower than what the Anthropic and Google adapters do
(they have no system role inside the conversation at all, so every system turn after the head reads
as a labelled user turn). **A system turn keeps its role exactly where a user turn follows it** (a
run of system turns counting as one, so ``developer, developer, user`` is kept whole): the brief, a
leading system prompt, a compaction summary at the head of the kept conversation. **Every other one
is sent as a user turn marked `SYSTEM_LABEL`**, so the model can still tell the harness's note from
a peer's words. That covers the trailing step note and also every earlier step note in the
transcript, which a later call finds followed by the assistant turn it preceded: a request then
holds only the orders measured as accepted, never ``developer, assistant``, which was not measured.
It also keeps a tool loop's prefix byte-stable: a note sent last is relabelled, and once the reply it
preceded follows it, it is still not followed by a user turn, so every later step sends it the same
way and a cached prefix holds.

The relabelling is a **copy, made at the wire and never persisted**: the transcript, the context
attribution line, compaction and every recovery reader still see a system turn, so nothing but the
request changes, and only for the model it is declared for.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from basecradle_harness._messages import Message

#: How a system turn reads where it travels as a user turn: the Anthropic and Google adapters'
#: label for a system turn after the head of the conversation, and the OpenAI wire's under
#: ``before_user``. One constant, so the model reads the harness's notes the same way on every wire.
SYSTEM_LABEL = "[System]\n"

#: ``AI_SYSTEM_TURNS``'s values. ``anywhere`` (the default, and what an unset or blank value reads
#: as) sends every system turn in the system role, wherever it falls: the wire as it has always been.
ANYWHERE = "anywhere"
BEFORE_USER = "before_user"
PLACEMENTS = (ANYWHERE, BEFORE_USER)


def before_user(messages: Sequence[Message]) -> list[Message]:
    """The conversation with every system turn not followed by a user turn sent as a labelled one.

    A run of consecutive system turns is judged as one: kept in the system role when the next turn
    that is not a system turn is a ``user`` turn, relabelled otherwise (an ``assistant`` or ``tool``
    turn next, or nothing at all). A relabelled turn is a copy, with every other field kept; the
    list handed in is never mutated, and every other turn is the same object.
    """
    placed = list(messages)
    following = None  # the role of the next turn that is not a system turn, walking backwards
    for index in range(len(placed) - 1, -1, -1):
        message = placed[index]
        if message.role != "system":
            following = message.role
        elif following != "user":
            placed[index] = replace(
                message, role="user", content=SYSTEM_LABEL + (message.content or "")
            )
    return placed
