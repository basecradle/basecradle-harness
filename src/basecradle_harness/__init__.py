"""Harness — a safe, modular agentic framework for BaseCradle.

A hackable reference you build *on*, not a black box: a small, readable agent
core with clean extension points for human AI developers to fork and extend.

https://basecradle.com · API docs: https://basecradle.com/docs/api
"""

from basecradle_harness._basecradle import TimelineAgent
from basecradle_harness._engine import Engine
from basecradle_harness._exceptions import (
    EngineError,
    HarnessError,
    PolicyError,
    ProviderAPIError,
    ProviderAuthError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
)
from basecradle_harness._harness import Harness
from basecradle_harness._memory import MemoryTool
from basecradle_harness._messages import Message, Role, ToolCall, ToolSpec
from basecradle_harness._openai import OpenAICompatibleProvider
from basecradle_harness._policy import SHELL, Policy
from basecradle_harness._provider import Provider
from basecradle_harness._session import Session
from basecradle_harness._tools import Tool, ToolRegistry
from basecradle_harness._version import __version__

__all__ = [
    "__version__",
    # The agent
    "Harness",
    "Session",
    "Engine",
    "TimelineAgent",
    # Provider contract + adapter
    "Provider",
    "OpenAICompatibleProvider",
    # Tools, registry, and the safety boundary
    "Tool",
    "ToolRegistry",
    "MemoryTool",
    "Policy",
    "SHELL",
    # Message vocabulary
    "Message",
    "Role",
    "ToolCall",
    "ToolSpec",
    # Errors
    "HarnessError",
    "PolicyError",
    "EngineError",
    "ProviderError",
    "ProviderConnectionError",
    "ProviderAPIError",
    "ProviderAuthError",
    "ProviderRateLimitError",
]
