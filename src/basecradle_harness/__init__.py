"""Harness — a safe, modular agentic framework for BaseCradle.

A hackable reference you build *on*, not a black box: a small, readable agent
core with clean extension points for human AI developers to fork and extend.

https://basecradle.com · API docs: https://basecradle.com/docs/api
"""

from basecradle_harness._assets import AssetsTool
from basecradle_harness._audio import HearAudioTool
from basecradle_harness._basecradle import TimelineAgent
from basecradle_harness._brief import (
    compose_brief,
    fetch_dashboard_md,
    render_defects,
    render_manifest,
    render_safety,
)
from basecradle_harness._code import CodeAttachTool, CodeExecutionBridge
from basecradle_harness._confirmed import ConfirmedTimelineAction
from basecradle_harness._delete import DeleteTool
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
from basecradle_harness._grok import (
    GrokEditImageTool,
    GrokGenerateImageTool,
    GrokGenerateVideoTool,
)
from basecradle_harness._harness import Harness
from basecradle_harness._images import EditImageTool, GenerateImageTool
from basecradle_harness._install import (
    InstallReport,
    charter_from_config,
    config_home,
    install,
    installed_version,
    prompt_text,
    reconcile_on_upgrade,
    system_prompt_text,
)
from basecradle_harness._lock import LockTool
from basecradle_harness._mcp import (
    HttpMcpClient,
    McpClient,
    McpError,
    McpResolution,
    McpServerConfig,
    McpTool,
    StdioMcpClient,
    load_mcp_configs,
    load_mcp_tools,
)
from basecradle_harness._memory import MemoryTool, SqliteMemoryStore
from basecradle_harness._memory_provider import (
    MemoryExchange,
    MemoryProvider,
    MemoryScope,
    SqliteMemoryProvider,
    memory_provider_from_env,
)
from basecradle_harness._messages import (
    CodeExecutionFile,
    CodeExecutionTrace,
    ImageContent,
    Message,
    Role,
    ToolCall,
    ToolResult,
    ToolSpec,
)
from basecradle_harness._openai import OpenAIProvider
from basecradle_harness._openrouter import OpenRouterProvider
from basecradle_harness._platform import (
    PlatformContext,
    PlatformTool,
    bind_platform_tools,
)
from basecradle_harness._plugins import (
    ActivationContext,
    EnvSet,
    LoadedPlugins,
    OpenAIKey,
    OpenAISurface,
    Requirement,
    ResolvedTools,
    ToolPlugin,
    Vendor,
    load_plugins,
    load_plugins_report,
    resolve_plugins,
)
from basecradle_harness._policy import BASECRADLE, SHELL, Policy
from basecradle_harness._provider import Provider
from basecradle_harness._reads import MessagesTool, UsersTool
from basecradle_harness._session import Session
from basecradle_harness._tasks import TasksTool
from basecradle_harness._tools import Tool, ToolRegistry
from basecradle_harness._version import __version__
from basecradle_harness._wake import (
    BreakerDecision,
    ClaimStore,
    MarkStore,
    ReadPacer,
    SeenStore,
    WakeAgent,
    WakeBreaker,
)
from basecradle_harness._webfetch import WebFetchTool
from basecradle_harness._webhooks import WebhookEndpointsTool, WebhookEventsTool
from basecradle_harness._xai_sdk import XaiSdkProvider

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
    "reconcile_on_upgrade",
    "installed_version",
    "config_home",
    "charter_from_config",
    "prompt_text",
    "system_prompt_text",
    "InstallReport",
    "MarkStore",
    "SeenStore",
    "ClaimStore",
    "WakeBreaker",
    "BreakerDecision",
    "ReadPacer",
    # The persistent Turn-0 operating brief
    "compose_brief",
    "render_manifest",
    "render_safety",
    "render_defects",
    "fetch_dashboard_md",
    # Provider contract + adapters
    "Provider",
    "OpenAIProvider",
    "XaiSdkProvider",
    "OpenRouterProvider",
    # Tools, registry, and the safety boundary
    "Tool",
    "ToolRegistry",
    "MemoryTool",
    "WebFetchTool",
    "Policy",
    # Pluggable memory: the provider seam (tools + store + observe/context hooks)
    "MemoryProvider",
    "SqliteMemoryProvider",
    "SqliteMemoryStore",
    "MemoryScope",
    "MemoryExchange",
    "memory_provider_from_env",
    "SHELL",
    "BASECRADLE",
    # Tool plugin framework: (name + requires + impl), provider-aware activation
    "ToolPlugin",
    "Requirement",
    "Vendor",
    "OpenAISurface",
    "EnvSet",
    "OpenAIKey",
    "ActivationContext",
    "ResolvedTools",
    "LoadedPlugins",
    "resolve_plugins",
    "load_plugins",
    "load_plugins_report",
    # MCP drop-in: the harness as an MCP client (safe-by-default, opt-out surfaced)
    "McpClient",
    "StdioMcpClient",
    "HttpMcpClient",
    "McpServerConfig",
    "McpTool",
    "McpResolution",
    "McpError",
    "load_mcp_configs",
    "load_mcp_tools",
    # Platform-aware tools (the SDK as tools)
    "PlatformTool",
    "PlatformContext",
    "ConfirmedTimelineAction",
    "AssetsTool",
    "HearAudioTool",
    "TasksTool",
    "TimelinesTool",
    "TrustTool",
    "LockTool",
    "DeleteTool",
    "UsersTool",
    "MessagesTool",
    "GenerateImageTool",
    "EditImageTool",
    "CodeExecutionBridge",
    "CodeAttachTool",
    "GrokEditImageTool",
    "GrokGenerateImageTool",
    "GrokGenerateVideoTool",
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
    "CodeExecutionTrace",
    "CodeExecutionFile",
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
