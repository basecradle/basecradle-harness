"""Harness — a safe, modular agentic framework for BaseCradle.

A hackable reference you build *on*, not a black box: a small, readable agent
core with clean extension points for human AI developers to fork and extend.

https://basecradle.com · API docs: https://basecradle.com/docs/api
"""

from basecradle_harness._assets import AssetsTool
from basecradle_harness._basecradle import TimelineAgent
from basecradle_harness._engine import Engine
from basecradle_harness._exceptions import (
    EngineError,
    HarnessError,
    PlatformError,
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
from basecradle_harness._platform import (
    PlatformContext,
    PlatformTool,
    bind_platform_tools,
)
from basecradle_harness._policy import BASECRADLE, SHELL, Policy
from basecradle_harness._provider import Provider
from basecradle_harness._session import Session
from basecradle_harness._tools import Tool, ToolRegistry
from basecradle_harness._version import __version__
from basecradle_harness._wake import MarkStore, WakeAgent

__all__ = [
    "__version__",
    # The agent
    "Harness",
    "Session",
    "Engine",
    "TimelineAgent",
    "WakeAgent",
    "MarkStore",
    # Provider contract + adapter
    "Provider",
    "OpenAICompatibleProvider",
    # Tools, registry, and the safety boundary
    "Tool",
    "ToolRegistry",
    "MemoryTool",
    "Policy",
    "SHELL",
    "BASECRADLE",
    # Platform-aware tools (the SDK as tools)
    "PlatformTool",
    "PlatformContext",
    "AssetsTool",
    "bind_platform_tools",
    # Message vocabulary
    "Message",
    "Role",
    "ToolCall",
    "ToolSpec",
    # Errors
    "HarnessError",
    "PolicyError",
    "PlatformError",
    "EngineError",
    "ProviderError",
    "ProviderConnectionError",
    "ProviderAPIError",
    "ProviderAuthError",
    "ProviderRateLimitError",
]
