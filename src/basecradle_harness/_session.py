"""A `Session`: one conversation thread on one input channel.

The unified-identity model, in one small file. An agent is a single
identity-and-memory locus, but it is addressed over many channels — a GitHub PR
thread, a BaseCradle timeline, any future input the router learns to forward.
Each channel is a *different conversation*, not one merged transcript; yet all of
them must share what the agent *knows* and *is*. (This is law: the BaseCradle
constitution, "Sovereignty and Governance" → identity is *unified* — "what
converges is memory and charter, not conversation.")

A `Session` is that one conversation: its own `history`, keyed by a `source`
string the caller chooses (`"github:pr-123"`, `"timeline:<uuid>"`, `"default"`).
What it does *not* own is the agent's brain, hands, or memory — those live on the
shared `Engine` (provider + tool registry, including the memory tool) it runs
against. So two sessions of the same agent keep separate transcripts but write to
and read from the *same* durable memory and start from the *same* charter. That
is "channels share memory, not conversation."

If given a `path`, a session persists its transcript there on every turn and
loads it on construction — so a past session's reasoning is readable after a
restart, the durable half of cross-session answerability. With no `path`, a
session is in-memory only (the default; transcripts of *live* sessions are still
readable from the one running instance via `Harness.transcript`).
"""

from __future__ import annotations

import json
from pathlib import Path

from basecradle_harness._engine import Engine
from basecradle_harness._messages import ImageContent, Message


class Session:
    """One channel's conversation, run against the agent's shared engine.

    Args:
        source: The channel/thread key this conversation belongs to. Free-form;
            the caller's convention (e.g. `"github:pr-123"`). It is the identity
            of the *conversation*, never of the agent.
        engine: The agent's shared loop — provider plus tool registry. Shared
            across every session of the agent, which is how separate transcripts
            still converge on one memory.
        system_prompt: The agent's charter, seeded as the first turn of a *new*
            session. A session reloaded from disk keeps its stored charter and is
            not reseeded.
        path: Where to persist this session's transcript. `None` (the default)
            keeps it in memory only. A path enables across-restart durability;
            its parent directory is created on first write.
    """

    def __init__(
        self,
        source: str,
        engine: Engine,
        *,
        system_prompt: str | None = None,
        path: str | Path | None = None,
    ) -> None:
        self.source = source
        self.engine = engine
        self.path = Path(path) if path is not None else None
        self.history: list[Message] = self._load()
        if not self.history and system_prompt:
            self.history.append(Message.system(system_prompt))

    def send(self, text: str, *, images: list[ImageContent] | None = None) -> str:
        """Send one user message, run the loop to a text reply, persist the turn.

        The full exchange is appended to `history` (and saved if this session has
        a path), so memory of *this* conversation carries into its next `send` —
        while the agent's durable memory, shared through the engine, carries
        across every conversation.

        `images` places pictures *in front of* the model on this turn (vision), so a
        peer's posted file is perceived directly rather than only described — the asset
        wake uses this. Once the model has answered, the pixels are **evicted** from the
        turn (the text stays as a breadcrumb): the same cost discipline the engine
        applies to a viewed image, so a presented picture is never re-sent (or re-billed)
        on a later turn, nor persisted as base64 into the transcript on disk.
        """
        turn = Message(role="user", content=text, images=list(images) if images else [])
        self.history.append(turn)
        try:
            reply = self.engine.run(self.history)
        finally:
            # Evict the pixels however the loop ends — including the error path — so a
            # failed turn cannot leave base64 in `history` to be re-sent (or persisted) on
            # a later `send`. Mirrors the engine's own finally-based image eviction. The
            # text stays as a breadcrumb; `_save` below only runs when the turn succeeded.
            turn.images = []
        self._save()
        return reply.content or ""

    # --- transcript persistence: load on construct, save on every turn --------

    def _load(self) -> list[Message]:
        if self.path is None or not self.path.exists():
            return []
        return [Message.from_dict(d) for d in json.loads(self.path.read_text())]

    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps([m.to_dict() for m in self.history], indent=2))
