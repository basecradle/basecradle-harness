"""The platform-aware tool seam: context, binding, and the policy capability.

`PlatformTool` is how a tool reaches the live SDK client and current timeline.
These tests pin the contract every Phase-2 tool depends on: an unbound tool fails
loudly-but-recoverably, binding wires only platform tools, and the `BASECRADLE`
capability is *permitted* by the shipped locked profile (platform I/O is the
point of a peer — only the shell is forbidden).
"""

import pytest

from basecradle_harness import (
    BASECRADLE,
    SHELL,
    Harness,
    MemoryTool,
    PlatformContext,
    PlatformError,
    PlatformTool,
    Policy,
    PolicyError,
    Tool,
    ToolRegistry,
    bind_platform_tools,
)

TIMELINE_UUID = "019e7750-66ee-7f53-829f-13a8a710b6da"
OTHER_TIMELINE = "019e7760-1234-7abc-8def-0123456789ab"


class Sentinel(PlatformTool):
    """A minimal platform tool that just reports the timeline it was bound to."""

    name = "sentinel"
    description = "Report the bound timeline."

    def run(self) -> str:
        return self.context.timeline


def test_platform_tool_requires_the_basecradle_capability():
    assert Sentinel().requires == frozenset({BASECRADLE})


def test_unbound_platform_tool_raises_a_clear_platform_error():
    tool = Sentinel()
    assert tool.bound is False
    with pytest.raises(PlatformError, match="not connected to one"):
        _ = tool.context


def test_bind_wires_the_context_and_run_can_read_it():
    tool = Sentinel()
    tool.bind(PlatformContext(client=object(), timeline=TIMELINE_UUID))
    assert tool.bound is True
    assert tool.run() == TIMELINE_UUID


def test_bind_platform_tools_binds_only_platform_tools_and_counts_them():
    platform = Sentinel()
    plain = MemoryTool()
    context = PlatformContext(client=object(), timeline=TIMELINE_UUID)

    bound = bind_platform_tools([platform, plain], context)

    assert bound == 1  # only the platform tool was bound
    assert platform.bound is True
    assert platform.run() == TIMELINE_UUID


def test_bind_platform_tools_walks_a_registry():
    registry = ToolRegistry()
    sentinel = registry.register(Sentinel())
    registry.register(MemoryTool())

    bound = bind_platform_tools(registry, PlatformContext(client=object(), timeline=TIMELINE_UUID))

    assert bound == 1
    assert sentinel.run() == TIMELINE_UUID


def test_locked_profile_permits_platform_tools():
    """A platform tool loads under the shipped safe profile — platform I/O is allowed."""
    Harness(_DummyProvider(), tools=[Sentinel()])  # no PolicyError


def test_basecradle_capability_is_not_among_the_forbidden_ones():
    assert BASECRADLE not in Policy.locked().forbidden
    assert SHELL in Policy.locked().forbidden


def test_a_profile_could_forbid_platform_io_if_it_wanted():
    """The capability is real: a profile that forbade it would refuse a platform tool."""
    no_platform = Policy(forbidden=frozenset({BASECRADLE}))
    with pytest.raises(PolicyError, match="basecradle"):
        ToolRegistry(policy=no_platform).register(Sentinel())


def test_platform_tool_is_still_a_tool():
    assert issubclass(PlatformTool, Tool)


# --- a no-op provider so a Harness can be built without a model --------------


class _DummyProvider:
    def chat(self, messages, tools=None):  # pragma: no cover - never called here
        raise AssertionError("not used")
