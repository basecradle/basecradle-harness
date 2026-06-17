"""Harness — a safe, modular agentic framework for BaseCradle.

A hackable reference you build *on*, not a black box: a small, readable agent
core with clean extension points for human AI developers to fork and extend.

https://basecradle.com · API docs: https://basecradle.com/docs/api
"""

from basecradle_harness._assets import AssetsTool
from basecradle_harness._audio import HearAudioTool
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
from basecradle_harness._governance import TimelinesTool, TrustTool
from basecradle_harness._harness import Harness
from basecradle_harness._images import GenerateImageTool
from basecradle_harness._install import (
    InstallReport,
    charter_from_config,
    config_home,
    install,
)
from basecradle_harness._memory import MemoryTool
from basecradle_harness._messages import (
    ImageContent,
    Message,
    Role,
    ToolCall,
    ToolResult,
    ToolSpec,
)
from basecradle_harness._openai import OpenAICompatibleProvider
from basecradle_harness._platform import (
    PlatformContext,
    PlatformTool,
    bind_platform_tools,
)
from basecradle_harness._policy import BASECRADLE, SHELL, Policy
from basecradle_harness._provider import Provider
from basecradle_harness._responses import OpenAIResponsesProvider
from basecradle_harness._session import Session
from basecradle_harness._tasks import TasksTool
from basecradle_harness._tools import Tool, ToolRegistry
from basecradle_harness._version import __version__
from basecradle_harness._wake import ClaimStore, MarkStore, SeenStore, WakeAgent
from basecradle_harness._webfetch import WebFetchTool
from basecradle_harness._webhooks import WebhookEndpointsTool, WebhookEventsTool

__all__ = [
    "__version__",
    # The agent
    "Harness",
    "Session",
    "Engine",
    "TimelineAgent",
    "WakeAgent",
    # Config home: installer + conffile upgrader
    "install",
    "config_home",
    "charter_from_config",
    "InstallReport",
    "MarkStore",
    "SeenStore",
    "ClaimStore",
    # Provider contract + adapters
    "Provider",
    "OpenAICompatibleProvider",
    "OpenAIResponsesProvider",
    # Tools, registry, and the safety boundary
    "Tool",
    "ToolRegistry",
    "MemoryTool",
    "WebFetchTool",
    "Policy",
    "SHELL",
    "BASECRADLE",
    # Platform-aware tools (the SDK as tools)
    "PlatformTool",
    "PlatformContext",
    "AssetsTool",
    "HearAudioTool",
    "TasksTool",
    "TimelinesTool",
    "TrustTool",
    "GenerateImageTool",
    "WebhookEndpointsTool",
    "WebhookEventsTool",
    "bind_platform_tools",
    # Message vocabulary
    "Message",
    "Role",
    "ToolCall",
    "ToolResult",
    "ToolSpec",
    "ImageContent",
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
