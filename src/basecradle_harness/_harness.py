"""`Harness` — the public front door: one agent, many conversations, one memory.

This is the class a developer imports. It wires the durable, *shared* pieces of
an agent — a profile (policy), a provider (the brain), a tool registry (the
hands, including memory), and the engine (the loop) — and then hands out
`Session`s on top of them, one per input channel.

That split is the unified-identity model the BaseCradle constitution requires: an
agent is *one* identity-and-memory locus addressed over many channels, and "what
converges is memory and charter, not conversation." A GitHub PR thread and a
BaseCradle timeline are different `Session`s with separate transcripts, but they
run against the *same* engine — so they share the agent's tools (hence the same
durable memory) and start from the same charter. They never bleed into one
incoherent transcript, yet they share what the agent *knows*.

Safe by default: the tool registry starts on `Policy.locked()`, so a tool that
needs a shell is refused the moment you try to add it — you never get a running
Harness that could reach a subprocess.

The simple case stays a one-liner — `send`/`history` operate on a default
session, so a single-channel agent never has to think about sessions at all:

    from basecradle_harness import Harness, OpenAIProvider, MemoryTool

    agent = Harness(
        OpenAIProvider(model="gpt-5.4-mini"),
        system_prompt="You are a helpful peer on BaseCradle.",
        tools=[MemoryTool()],
    )
    print(agent.send("Remember that my city is Dallas."))
    print(agent.send("What city am I in?"))

A multi-channel agent names each channel; the memory written on one is readable
from another, and a past session's transcript stays readable from any other:

    agent.send("I shipped the retry fix.", source="github:pr-123")
    agent.send("Why did you do that?", source="timeline:abc")   # same memory behind both
    agent.transcript("github:pr-123")                           # the PR session, readable here
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from urllib.parse import quote

from basecradle_harness._engine import DEFAULT_MAX_STEPS, Engine
from basecradle_harness._messages import Message
from basecradle_harness._policy import Policy
from basecradle_harness._provider import Provider
from basecradle_harness._session import Session
from basecradle_harness._tools import Tool, ToolRegistry

#: The session every plain `send`/`history` call uses, so a single-channel agent
#: never has to name a source.
DEFAULT_SOURCE = "default"


class Harness:
    """An agent: shared provider, tools, charter, and engine — sessions on top.

    Args:
        provider: The model backing the agent. Shared across all its sessions.
        system_prompt: An optional charter — the standing instructions seeded as
            the first turn of every new session.
        tools: Tools to register. Each is gated by `policy` as it is added; a
            forbidden tool raises `PolicyError` here, at construction. The tool
            instances are shared across sessions, so a stateful tool like
            `MemoryTool` *is* the agent's one durable memory.
        policy: The profile. Defaults to `Policy.locked()` — the safe Harness
            profile. Pass `Policy.unlocked()` only with intent.
        max_steps: The engine's per-turn provider-call budget.
        home: An optional directory under which session transcripts persist
            (`<home>/sessions/<source>.json`), making a past session's reasoning
            readable across restarts. `None` (the default) keeps sessions in
            memory only — still readable across sessions of the one running
            instance via `transcript`, just not across a restart.
    """

    def __init__(
        self,
        provider: Provider,
        *,
        system_prompt: str | None = None,
        tools: Iterable[Tool] | None = None,
        policy: Policy | None = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        home: str | Path | None = None,
    ) -> None:
        self.provider = provider
        self.tools = ToolRegistry(policy=policy or Policy.locked())
        for tool in tools or ():
            self.tools.register(tool)
        self.engine = Engine(provider, self.tools, max_steps=max_steps)
        self.system_prompt = system_prompt
        self.home = Path(home) if home is not None else None
        self._sessions: dict[str, Session] = {}

    def session(self, source: str = DEFAULT_SOURCE) -> Session:
        """Get (or create) the conversation for `source`.

        Every session of this agent runs against the same engine — the same
        provider, the same tools, the same memory — and is seeded from the same
        charter, so they converge on one identity while keeping separate
        transcripts.
        """
        existing = self._sessions.get(source)
        if existing is not None:
            return existing
        session = Session(
            source,
            self.engine,
            system_prompt=self.system_prompt,
            path=self._transcript_path(source),
        )
        self._sessions[source] = session
        return session

    def send(self, text: str, *, source: str = DEFAULT_SOURCE) -> str:
        """Send one user message to a channel's session and return the reply.

        With no `source` this is the simple single-conversation agent. Pass a
        `source` to address a specific channel; whatever the channel, the reply is
        produced against the agent's one shared memory.
        """
        return self.session(source).send(text)

    def transcript(self, source: str) -> list[Message]:
        """Read another session's transcript — the cross-session answerability seam.

        Returns a copy of the conversation `source` held, so reasoning done on one
        channel is reachable from another (e.g. answering on a timeline about work
        done in a GitHub session). A live session is read from memory; an idle one
        is loaded from its persisted transcript if this agent has a `home`. An
        unknown source returns an empty list.
        """
        live = self._sessions.get(source)
        if live is not None:
            return list(live.history)
        path = self._transcript_path(source)
        if path is not None and path.exists():
            return Session(source, self.engine, path=path).history
        return []

    @property
    def sessions(self) -> Mapping[str, Session]:
        """The live sessions, keyed by source. A read-only snapshot."""
        return dict(self._sessions)

    @property
    def history(self) -> list[Message]:
        """The default session's transcript — the single-channel agent's history."""
        return self.session(DEFAULT_SOURCE).history

    def _transcript_path(self, source: str) -> Path | None:
        """Where `source`'s transcript persists, or `None` if persistence is off.

        The source is percent-encoded into a single safe filename, so a key with
        a `:` or `/` (`"github:pr-123"`) maps to one file without colliding.
        """
        if self.home is None:
            return None
        return self.home / "sessions" / f"{quote(source, safe='')}.json"
