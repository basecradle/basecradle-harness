"""`Harness` — the public front door: a model, some tools, a safe profile.

This is the class a developer imports. It wires the four pieces together — a
profile (policy), a provider (the brain), a tool registry (the hands), and the
engine (the loop) — and gives back one method, `send`, that takes a line of text
and returns the agent's reply.

Safe by default: the tool registry starts on `Policy.locked()`, so a tool that
needs a shell is refused the moment you try to add it — you never get a running
Harness that could reach a subprocess.

    from basecradle_harness import Harness, OpenAICompatibleProvider, MemoryTool

    agent = Harness(
        OpenAICompatibleProvider(model="gpt-4o"),
        system_prompt="You are a helpful peer on BaseCradle.",
        tools=[MemoryTool()],
    )
    print(agent.send("Remember that my city is Dallas."))
    print(agent.send("What city am I in?"))
"""

from __future__ import annotations

from collections.abc import Iterable

from basecradle_harness._engine import DEFAULT_MAX_STEPS, Engine
from basecradle_harness._messages import Message
from basecradle_harness._policy import Policy
from basecradle_harness._provider import Provider
from basecradle_harness._tools import Tool, ToolRegistry


class Harness:
    """An agent: a provider, a tool registry on a policy, and the loop over them.

    Args:
        provider: The model backing the agent.
        system_prompt: An optional first `system` message, seeding the agent's
            standing instructions.
        tools: Tools to register. Each is gated by `policy` as it is added; a
            forbidden tool raises `PolicyError` here, at construction.
        policy: The profile. Defaults to `Policy.locked()` — the safe Harness
            profile. Pass `Policy.unlocked()` only with intent.
        max_steps: The engine's per-turn provider-call budget.
    """

    def __init__(
        self,
        provider: Provider,
        *,
        system_prompt: str | None = None,
        tools: Iterable[Tool] | None = None,
        policy: Policy | None = None,
        max_steps: int = DEFAULT_MAX_STEPS,
    ) -> None:
        self.provider = provider
        self.tools = ToolRegistry(policy=policy or Policy.locked())
        for tool in tools or ():
            self.tools.register(tool)
        self.engine = Engine(provider, self.tools, max_steps=max_steps)
        self.history: list[Message] = []
        if system_prompt:
            self.history.append(Message.system(system_prompt))

    def send(self, text: str) -> str:
        """Send one user message and run the loop to the agent's text reply.

        The full exchange — the user turn, the model's turns, and any tool
        results — is appended to `history`, so memory of the conversation carries
        into the next `send`.
        """
        self.history.append(Message.user(text))
        reply = self.engine.run(self.history)
        return reply.content or ""
