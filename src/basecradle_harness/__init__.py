"""Harness — a safe, modular agentic framework for BaseCradle.

A hackable reference you build *on*, not a black box: a small, readable agent
core with clean extension points for human AI developers to fork and extend.

https://basecradle.com · API docs: https://basecradle.com/docs/api
"""

from basecradle_harness._exceptions import (
    HarnessError,
    ProviderAPIError,
    ProviderAuthError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
)
from basecradle_harness._messages import Message, Role, ToolCall, ToolSpec
from basecradle_harness._openai import OpenAICompatibleProvider
from basecradle_harness._provider import Provider
from basecradle_harness._version import __version__

__all__ = [
    "__version__",
    # Provider contract + adapter
    "Provider",
    "OpenAICompatibleProvider",
    # Message vocabulary
    "Message",
    "Role",
    "ToolCall",
    "ToolSpec",
    # Errors
    "HarnessError",
    "ProviderError",
    "ProviderConnectionError",
    "ProviderAPIError",
    "ProviderAuthError",
    "ProviderRateLimitError",
]
